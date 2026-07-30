# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path

import pytest
import torch

from rlinf.envs.robotwin.seed_utils import partition_success_seeds


def _first_eval_seeds(
    *,
    seed_count: int,
    total_num_envs: int,
    total_num_processes: int,
    group_size: int = 1,
    base_seed: int = 0,
) -> list[int]:
    success_seeds = torch.arange(seed_count, dtype=torch.long)
    selected_seeds = []
    for seed_offset in range(total_num_processes):
        num_envs = total_num_envs // total_num_processes
        num_group = num_envs // group_size
        worker_seeds = partition_success_seeds(
            success_seeds,
            base_seed=base_seed,
            seed_offset=seed_offset,
            total_num_processes=total_num_processes,
            num_group=num_group,
        )
        selected_seeds.extend(worker_seeds[:num_group].tolist())
    return selected_seeds


@pytest.mark.parametrize(
    ("seed_count", "total_num_envs", "total_num_processes"),
    [
        (320, 128, 4),
        (320, 128, 8),
        (320, 128, 16),
        (260, 128, 4),
        (200, 128, 8),
        (150, 128, 4),
    ],
)
def test_robotwin_eval_success_seeds_do_not_overlap_across_workers(
    seed_count: int,
    total_num_envs: int,
    total_num_processes: int,
):
    """Regression test for duplicate RoboTwin eval seeds across EnvWorkers."""
    selected_seeds = _first_eval_seeds(
        seed_count=seed_count,
        total_num_envs=total_num_envs,
        total_num_processes=total_num_processes,
    )

    assert len(selected_seeds) == total_num_envs
    assert len(set(selected_seeds)) == total_num_envs


def test_robotwin_eval_success_seed_order_is_controlled_by_base_seed():
    selected_seed_0 = _first_eval_seeds(
        seed_count=320,
        total_num_envs=128,
        total_num_processes=4,
        base_seed=0,
    )
    selected_seed_0_again = _first_eval_seeds(
        seed_count=320,
        total_num_envs=128,
        total_num_processes=4,
        base_seed=0,
    )
    selected_seed_1 = _first_eval_seeds(
        seed_count=320,
        total_num_envs=128,
        total_num_processes=4,
        base_seed=1,
    )

    assert selected_seed_0 == selected_seed_0_again
    assert selected_seed_0 != selected_seed_1


def _periodic_eval_events(
    success_seeds: torch.Tensor,
    *,
    total_num_envs: int,
    total_num_processes: int,
    rollout_epoch: int,
    event_count: int,
    base_seed: int = 0,
) -> tuple[list[list[int]], list[set[int]]]:
    """Simulate consecutive explicit-reset eval events across EnvWorkers."""
    selected_events = [[] for _ in range(event_count)]
    worker_seed_sets = []
    num_envs = total_num_envs // total_num_processes
    for seed_offset in range(total_num_processes):
        worker_seeds = partition_success_seeds(
            success_seeds,
            base_seed=base_seed,
            seed_offset=seed_offset,
            total_num_processes=total_num_processes,
            num_group=num_envs,
        )
        current_seed_index = 0
        for event_index in range(event_count):
            worker_event = []
            for _ in range(rollout_epoch):
                indices = (
                    torch.arange(num_envs, device=worker_seeds.device)
                    + current_seed_index
                ) % worker_seeds.numel()
                worker_event.extend(worker_seeds[indices].tolist())
                current_seed_index = (
                    current_seed_index + num_envs
                ) % worker_seeds.numel()
            selected_events[event_index].extend(worker_event)
            if event_index == 0:
                worker_seed_sets.append(set(worker_event))
    return selected_events, worker_seed_sets


def test_robotwin_rlt_periodic_eval_reuses_exact_20_seed_bank():
    repo_root = Path(__file__).resolve().parents[2]
    seed_bank_path = (
        repo_root
        / "rlinf"
        / "envs"
        / "robotwin"
        / "seeds"
        / "eval_seeds_adjust_bottle_rlt_periodic20_v1.json"
    )
    with seed_bank_path.open(encoding="utf-8") as f:
        seed_bank = json.load(f)["adjust_bottle"]["success_seeds"]

    assert len(seed_bank) == 20
    assert len(set(seed_bank)) == 20

    success_seeds = torch.as_tensor(seed_bank, dtype=torch.long)
    events, worker_sets = _periodic_eval_events(
        success_seeds,
        total_num_envs=4,
        total_num_processes=2,
        rollout_epoch=5,
        event_count=2,
    )
    event_1, event_2 = events

    assert len(event_1) == 20
    assert len(set(event_1)) == 20
    assert set(event_1) == set(seed_bank)
    assert event_2 == event_1
    assert worker_sets[0].isdisjoint(worker_sets[1])
