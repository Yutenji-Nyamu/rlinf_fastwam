# Copyright 2025 The RLinf Authors.
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

"""CPU integration checks of actual actor methods, without FSDP/Ray imports.

Run in the source-locked server environment. The AST harness changes only the
base class and strips scheduling decorators; method bodies come from the actor.
"""

import ast
import asyncio
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import rlinf.algorithms.dvac_two_level as helper_module
from rlinf.algorithms.dvac_two_level import compute_dvac_two_level_weights


class CheckpointBase:
    def save_checkpoint(self, save_path, step=0):
        Path(save_path).mkdir(parents=True, exist_ok=True)
        self.base_saved_step = step

    def load_checkpoint(self, load_path):
        self.base_loaded = load_path


def actor_harness():
    path = (
        Path(helper_module.__file__).resolve().parents[1]
        / "workers/actor/embodied_fsdp_actor_worker.py"
    )
    module = ast.parse(path.read_text(encoding="utf-8"))
    actor = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "EmbodiedFSDPActor"
    )
    wanted = {
        "_dvac_two_level_contract",
        "_prepare_dvac_two_level_step",
        "_prepare_dvac_train_step",
        "_write_dvac_step_artifact",
        "_dvac_sidecar_path",
        "save_checkpoint",
        "load_checkpoint",
        "recv_rollout_trajectories",
    }
    methods = [
        node
        for node in actor.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in wanted
    ]
    assert {node.name for node in methods} == wanted
    for method in methods:
        # Retain the actual no-grad contract, remove only worker timers.
        method.decorator_list = [
            node for node in method.decorator_list if "no_grad" in ast.unparse(node)
        ]
    harness = ast.ClassDef(
        name="ActorHarness",
        bases=[ast.Name(id="CheckpointBase", ctx=ast.Load())],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    extracted = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            harness,
        ],
        type_ignores=[],
    )
    namespace = {
        "torch": torch,
        "Path": Path,
        "json": json,
        "os": os,
        "CheckpointBase": CheckpointBase,
        "compute_dvac_two_level_weights": compute_dvac_two_level_weights,
        "clear_memory": lambda **kwargs: None,
        "compute_split_num": lambda send, recv: 1,
        "convert_trajectories_to_batch": lambda packets: packets[0].batch,
    }
    exec(compile(ast.fix_missing_locations(extracted), str(path), "exec"), namespace)
    return namespace["ActorHarness"]


def sample_shards():
    shards = []
    for rank in range(2):
        variance = (torch.arange(36).reshape(3, 4, 3).float() / 13 + rank).exp()
        mask = torch.tensor(
            [[1, 1, 1, 1], [1, 1, 0, 1], [0, 1, 0, 0]], dtype=torch.bool
        )
        if rank:
            mask = mask.roll(1, dims=1)
        variance[~mask] = float("nan")
        adv = (
            torch.tensor([1.0, -1.0, 0.0, 2.0])[None, :, None].expand(3, -1, -1).clone()
        )
        adv[~mask] = 0
        lengths = (
            torch.tensor([60.0, 100.0, 150.0, 200.0])[None, :, None]
            .expand(3, -1, -1)
            .clone()
        )
        lengths[~mask] = 0
        ids = torch.full((4,), rank + 10, dtype=torch.int64)
        shards.append((variance, mask, adv, lengths, ids))
    return shards


def make_actor(rank=0, scope="both", tmp_path=None):
    actor = actor_harness()()
    actor._rank, actor._world_size, actor.version = rank, 2, 7
    actor.device = torch.device("cpu")
    actor.dvac_selected_l = 3
    actor.dvac_train_enabled = actor.dvac_two_level_enabled = True
    actor.dvac_train_mode = "apply"
    actor.dvac_train_application = "chunk_clipped_action_advantage"
    actor.dvac_train_cfg = {
        "normalization": "two_level_group",
        "scope": scope,
        "alpha_local": 0.7,
        "alpha_chunk": 0.8,
        "log_eps": 1e-12,
        "minmax_eps": 1e-6,
        "save_step_tensors": True,
    }
    actor.cfg = SimpleNamespace(
        algorithm=SimpleNamespace(group_size=4),
        env=SimpleNamespace(train=SimpleNamespace(max_episode_steps=200)),
    )
    actor.stage_num = 1
    actor._component_placement = SimpleNamespace(get_world_size=lambda component: 2)
    actor._process_received_rollout_batch = lambda batch: batch
    actor._dvac_pending_step = None
    actor.dvac_output_dir = tmp_path
    actor.dvac_recent_stats = None
    actor._new_dvac_recent_stats = lambda: pytest.fail(
        "Two-level mode touched historical statistics"
    )
    variance, mask, adv, lengths, ids = sample_shards()[rank]
    actor.rollout_batch = {
        "forward_inputs": {"dvac_v_l3": variance.clone().requires_grad_()},
        "advantages": adv.clone().requires_grad_(),
        "loss_mask": mask,
        "loss_mask_sum": lengths,
    }
    actor._dvac_rollout_group_ids = ids
    return actor


def gather_packets(shards):
    fields = []
    fields.append([torch.tensor(shard[0].shape) for shard in shards])
    fields.append([shard[0] for shard in shards])
    fields.append([shard[1].unsqueeze(-1) for shard in shards])
    fields.append([shard[2] for shard in shards])
    fields.append(
        [200 / torch.where(shard[1].unsqueeze(-1), shard[3], 1.0) for shard in shards]
    )
    fields.append([shard[4] for shard in shards])
    return fields


def expected_weights(scope="both", shards=None):
    packets = gather_packets(shards or sample_shards())
    return compute_dvac_two_level_weights(
        torch.cat(packets[1], 1),
        torch.cat(packets[2], 1),
        torch.cat(packets[5], 0),
        torch.cat(packets[3], 1),
        chunk_contributions=torch.cat(packets[4], 1).squeeze(-1),
        scope=scope,
        alpha_local=0.7,
        alpha_chunk=0.8,
    )[0]


def fake_all_gather(monkeypatch, actor, packets=None):
    packets = packets or gather_packets(sample_shards())
    calls = []

    def gather(output, value):
        parts = packets[len(calls)]
        torch.testing.assert_close(value, parts[actor._rank], equal_nan=True)
        for target, source in zip(output, parts):
            target.copy_(source)
        calls.append(value.shape)

    monkeypatch.setattr(torch.distributed, "all_gather", gather)
    return calls


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("scope", ["both", "positive", "negative"])
def test_actual_prepare_gathers_then_freezes_native_chunk_weights(
    monkeypatch, tmp_path, rank, scope
):
    actor = make_actor(rank, scope, tmp_path)
    calls = fake_all_gather(monkeypatch, actor)
    original = actor.rollout_batch["forward_inputs"]["dvac_v_l3"]
    actor._prepare_dvac_train_step()
    weights = actor.rollout_batch["forward_inputs"]["dvac_weights"]
    expected = expected_weights(scope)[:, rank * 4 : (rank + 1) * 4]
    torch.testing.assert_close(weights, expected)
    assert weights.shape == (3, 4, 3) and not weights.requires_grad
    assert len(calls) == 6
    assert "dvac_v_l3" not in actor.rollout_batch["forward_inputs"]
    assert actor.dvac_recent_stats is None
    pending = actor._dvac_pending_step
    assert pending["group_ids"].shape == (4,)
    torch.testing.assert_close(pending["group_ids"], actor._dvac_rollout_group_ids)
    torch.testing.assert_close(pending["weights"], expected)
    assert pending["loss_mask"].shape == (3, 4, 1)
    assert pending["advantages"].shape == (3, 4, 1)
    assert pending["metrics"]["actor/dvac_group_count"] == 2
    assert pending["metrics"]["actor/dvac_weight_mean"] == pytest.approx(1.0, abs=2e-6)
    with torch.no_grad():
        original.add_(5.0)
    torch.testing.assert_close(weights, expected)
    # The actor has consumed the rollout signal; optimizer epochs cannot silently
    # recompute weights from updated model state or historical statistics.
    with pytest.raises(ValueError, match="Missing rollout DVAC signal"):
        actor._prepare_dvac_train_step()


def test_prepare_rejects_incomplete_global_group_ids(monkeypatch):
    actor = make_actor()
    packets = gather_packets(sample_shards())
    packets[-1][1] = torch.tensor([11, 11, 11, 12])
    fake_all_gather(monkeypatch, actor, packets)
    with pytest.raises(ValueError, match="complete native GRPO groups"):
        actor._prepare_dvac_train_step()


def test_prepare_rejects_mismatched_native_shards_before_tensor_gather(monkeypatch):
    actor = make_actor()
    packets = gather_packets(sample_shards())
    packets[0][1] = torch.tensor([4, 4, 3])
    calls = fake_all_gather(monkeypatch, actor, packets)
    with pytest.raises(ValueError, match="equal-sized native actor shards"):
        actor._prepare_dvac_train_step()
    assert len(calls) == 1


def test_prepare_rejects_nonpositive_active_native_denominator():
    actor = make_actor()
    actor.rollout_batch["loss_mask_sum"][0, 0, 0] = 0
    with pytest.raises(ValueError, match="Invalid native loss_mask_sum"):
        actor._prepare_dvac_train_step()


class FakeChannel:
    def __init__(self, packet):
        self.packet = packet

    def get(self, async_op=False):
        return self

    async def async_wait(self):
        return self.packet


def test_recv_freezes_native_group_ids_and_rejects_split_packet():
    actor = make_actor(rank=1)
    packet = SimpleNamespace(rewards=torch.zeros(3, 8, 1))
    packet.batch = {"rewards": packet.rewards}
    asyncio.run(actor.recv_rollout_trajectories(FakeChannel(packet)))
    torch.testing.assert_close(
        actor._dvac_rollout_group_ids, torch.tensor([2, 2, 2, 2, 3, 3, 3, 3])
    )
    packet.rewards = torch.zeros(3, 6, 1)
    with pytest.raises(
        ValueError, match="complete GRPO groups in each received packet"
    ):
        asyncio.run(actor.recv_rollout_trajectories(FakeChannel(packet)))


def test_step_sidecar_and_resume_lock_two_level_contract(monkeypatch, tmp_path):
    actor = make_actor(tmp_path=tmp_path)
    fake_all_gather(monkeypatch, actor)
    actor._prepare_dvac_train_step()
    actor._write_dvac_step_artifact(actor._dvac_pending_step)
    step_path = tmp_path / "runner_step_0007.pt"
    saved = torch.load(step_path, weights_only=False)
    assert saved["schema_version"] == 2 and saved["actor_rank"] == 0
    torch.testing.assert_close(saved["weights"], actor._dvac_pending_step["weights"])
    assert saved["group_ids"].shape == (4,)
    assert not (tmp_path / "runner_step_0007.partial.pt").exists()
    with pytest.raises(FileExistsError):
        actor._write_dvac_step_artifact(actor._dvac_pending_step)
    with pytest.raises(RuntimeError, match="partially applied"):
        actor.save_checkpoint(str(tmp_path / "checkpoint"), 7)
    actor._dvac_pending_step = None
    actor.save_checkpoint(str(tmp_path / "checkpoint"), 7)
    sidecar = actor._dvac_sidecar_path(str(tmp_path / "checkpoint"))
    data = json.loads(sidecar.read_text())
    assert data["recent_stats"] is None
    assert data["two_level_config"] == actor._dvac_two_level_contract()
    actor.load_checkpoint(str(tmp_path / "checkpoint"))
    assert actor.dvac_recent_stats is None
    actor.dvac_train_cfg["alpha_chunk"] = 0.1
    with pytest.raises(ValueError, match="two-level resume configuration mismatch"):
        actor.load_checkpoint(str(tmp_path / "checkpoint"))


def _gloo_actor_worker(rank, world_size, directory):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=(Path(directory) / "gloo_init").as_uri(),
        timeout=timedelta(seconds=45),
    )
    try:
        actor = make_actor(rank=rank)
        actor._prepare_dvac_train_step()
        weights = actor.rollout_batch["forward_inputs"]["dvac_weights"]
        torch.testing.assert_close(
            weights, expected_weights()[:, rank * 4 : (rank + 1) * 4]
        )
        torch.save(weights, Path(directory) / f"rank{rank}.pt")
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="CPU gloo unavailable",
)
def test_real_two_rank_gloo_prepare_matches_full_rollout(tmp_path):
    torch.multiprocessing.spawn(
        _gloo_actor_worker, args=(2, str(tmp_path)), nprocs=2, join=True
    )
    gathered = torch.cat(
        [
            torch.load(tmp_path / f"rank{rank}.pt", weights_only=True)
            for rank in range(2)
        ],
        dim=1,
    )
    torch.testing.assert_close(gathered, expected_weights())
