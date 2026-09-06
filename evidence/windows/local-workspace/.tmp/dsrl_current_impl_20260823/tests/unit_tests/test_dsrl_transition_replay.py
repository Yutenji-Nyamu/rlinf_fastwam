# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from rlinf.data.schema.embodied_types import Trajectory
from rlinf.data.storage.replay import (
    DSRLTransitionReplayBuffer,
    project_dsrl_trajectory,
)


def _make_robotwin_trajectory() -> Trajectory:
    trajectory_length = 3
    batch_size = 2
    action_horizon = 50
    latent_dim = 32
    chunk_size = 20

    latent = torch.arange(
        trajectory_length * batch_size * latent_dim, dtype=torch.bfloat16
    ).reshape(trajectory_length, batch_size, latent_dim)
    actions = latent.unsqueeze(2).repeat(1, 1, action_horizon, 1)

    # Done fields contain one leading bootstrap slot for rollout_epoch=1.
    terminations = torch.zeros(
        trajectory_length + 1, batch_size, chunk_size, dtype=torch.bool
    )
    truncations = torch.zeros_like(terminations)
    terminations[2, 0, 7] = True
    truncations[2, 0, 7] = True
    terminations[3, 0] = True
    truncations[3, 1, -1] = True

    images = torch.arange(
        trajectory_length * batch_size * 72 * 96 * 3, dtype=torch.int64
    )
    images = (
        images.remainder(256)
        .to(torch.uint8)
        .reshape(trajectory_length, batch_size, 72, 96, 3)
    )
    states = torch.arange(
        trajectory_length * batch_size * 14, dtype=torch.float32
    ).reshape(trajectory_length, batch_size, 14)

    return Trajectory(
        actions=actions,
        rewards=torch.zeros(
            trajectory_length, batch_size, chunk_size, dtype=torch.float32
        ),
        terminations=terminations,
        truncations=truncations,
        dones=terminations | truncations,
        curr_obs={"main_images": images, "states": states},
        next_obs={"main_images": torch.flip(images, dims=(2,)), "states": states + 1},
    )


def test_robotwin_projection_contract():
    pytest.importorskip("openpi")
    batch = project_dsrl_trajectory(
        _make_robotwin_trajectory(),
        action_horizon=50,
        latent_dim=32,
        state_dim=14,
        num_action_chunks=20,
        gamma=0.999,
    )

    assert batch["actions"].shape == (5, 32)
    assert batch["curr_obs"]["main_images"].shape == (5, 3, 64, 64)
    assert batch["curr_obs"]["main_images"].dtype == torch.bfloat16
    assert batch["curr_obs"]["states"].shape == (5, 14)
    assert batch["rewards"].flatten().tolist() == [-1.0, -1.0, 0.0, -1.0, -1.0]
    assert batch["terminations"].flatten().tolist() == [
        False,
        False,
        True,
        False,
        False,
    ]
    assert batch["truncations"].flatten().tolist() == [
        False,
        False,
        True,
        False,
        True,
    ]
    assert batch["continuations"].flatten().tolist() == [
        True,
        True,
        False,
        True,
        True,
    ]
    assert torch.allclose(batch["discounts"], torch.full((5, 1), 0.999**20))


def _synthetic_batch(size: int) -> dict[str, object]:
    return {
        "curr_obs": {
            "main_images": torch.arange(size * 3 * 4 * 4, dtype=torch.float32).reshape(
                size, 3, 4, 4
            ),
            "states": torch.arange(size * 14, dtype=torch.float32).reshape(size, 14),
        },
        "next_obs": {
            "main_images": torch.arange(size * 3 * 4 * 4, dtype=torch.float32).reshape(
                size, 3, 4, 4
            )
            + 1,
            "states": torch.arange(size * 14, dtype=torch.float32).reshape(size, 14)
            + 1,
        },
        "actions": torch.arange(size * 32, dtype=torch.float32).reshape(size, 32),
        "rewards": -torch.ones(size, 1),
        "continuations": torch.ones(size, 1, dtype=torch.bool),
        "terminations": torch.zeros(size, 1, dtype=torch.bool),
        "truncations": torch.zeros(size, 1, dtype=torch.bool),
        "discounts": torch.full((size, 1), 0.999**20),
    }


def test_ring_replacement_sampling_and_rng_resume(tmp_path):
    replay = DSRLTransitionReplayBuffer(
        capacity=5,
        seed=17,
        rank=0,
        world_size=1,
    )
    assert replay.add_batch(_synthetic_batch(7)) == 7
    assert len(replay) == 5
    sampled = replay.sample(8)
    assert sampled["actions"].shape == (8, 32)

    save_path = tmp_path / "replay"
    replay.save_checkpoint(str(save_path))
    expected_next = replay.sample(8)

    restored = DSRLTransitionReplayBuffer(
        capacity=5,
        seed=999,
        rank=0,
        world_size=1,
    )
    restored.load_checkpoint(str(save_path))
    actual_next = restored.sample(8)

    assert restored.total_inserted == replay.total_inserted
    assert restored.write_cursor == replay.write_cursor
    assert restored.resident_size == replay.resident_size
    assert torch.equal(actual_next["actions"], expected_next["actions"])
    assert torch.equal(
        actual_next["curr_obs"]["main_images"],
        expected_next["curr_obs"]["main_images"],
    )


def test_ring_rejects_world_size_change(tmp_path):
    replay = DSRLTransitionReplayBuffer(
        capacity=6,
        seed=3,
        rank=0,
        world_size=1,
    )
    replay.add_batch(_synthetic_batch(2))
    save_path = tmp_path / "replay"
    replay.save_checkpoint(str(save_path))

    incompatible = DSRLTransitionReplayBuffer(
        capacity=6,
        seed=3,
        rank=0,
        world_size=2,
    )
    with pytest.raises(ValueError, match="layout mismatch"):
        incompatible.load_checkpoint(str(save_path))


def test_projection_rejects_non_repeated_latent():
    pytest.importorskip("openpi")
    trajectory = _make_robotwin_trajectory()
    trajectory.actions[0, 0, 1, 0] += 1
    with pytest.raises(ValueError, match="not exactly repeated"):
        project_dsrl_trajectory(
            trajectory,
            action_horizon=50,
            latent_dim=32,
            state_dim=14,
            num_action_chunks=20,
            gamma=0.999,
        )
