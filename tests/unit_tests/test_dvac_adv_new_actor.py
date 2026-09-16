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
from rlinf.algorithms.dvac_linear_controls import (
    apply_chunk_dropout,
    effective_linear_alphas,
    linear_controls_contract,
)
from rlinf.algorithms.dvac_two_level import (
    canonicalize_dvac_two_level_contract,
    compute_dvac_two_level_weights,
    dvac_mapping_contract,
)


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
        "dvac_mapping_contract": dvac_mapping_contract,
        "canonicalize_dvac_two_level_contract": canonicalize_dvac_two_level_contract,
        "apply_chunk_dropout": apply_chunk_dropout,
        "effective_linear_alphas": effective_linear_alphas,
        "linear_controls_contract": linear_controls_contract,
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


def make_actor(rank=0, scope="both", tmp_path=None, mapping="linear_centered"):
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
    if mapping == "exp_mean":
        actor.dvac_train_cfg.update(
            mapping=mapping, temperature_local=0.5, temperature_chunk=0.5
        )
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


def expected_weights(
    scope="both",
    shards=None,
    mapping="linear_centered",
    alpha_local=0.7,
    alpha_chunk=0.8,
):
    packets = gather_packets(shards or sample_shards())
    return compute_dvac_two_level_weights(
        torch.cat(packets[1], 1),
        torch.cat(packets[2], 1),
        torch.cat(packets[5], 0),
        torch.cat(packets[3], 1),
        chunk_contributions=torch.cat(packets[4], 1).squeeze(-1),
        scope=scope,
        alpha_local=alpha_local,
        alpha_chunk=alpha_chunk,
        mapping=mapping,
        temperature_local=0.5,
        temperature_chunk=0.5,
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
@pytest.mark.parametrize("mapping", ["linear_centered", "exp_mean"])
def test_actual_prepare_gathers_then_freezes_native_chunk_weights(
    monkeypatch, tmp_path, rank, scope, mapping
):
    actor = make_actor(rank, scope, tmp_path, mapping)
    calls = fake_all_gather(monkeypatch, actor)
    original = actor.rollout_batch["forward_inputs"]["dvac_v_l3"]
    actor._prepare_dvac_train_step()
    weights = actor.rollout_batch["forward_inputs"]["dvac_weights"]
    expected = expected_weights(scope, mapping=mapping)[:, rank * 4 : (rank + 1) * 4]
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
    assert pending["config"] == actor._dvac_two_level_contract()
    if mapping == "exp_mean":
        assert pending["config"]["mapping"] == "exp_mean"
        assert pending["config"]["temperature_local"] == 0.5
        assert pending["config"]["temperature_chunk"] == 0.5
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


@pytest.mark.parametrize("mapping", ["linear_centered", "exp_mean"])
def test_step_sidecar_and_resume_lock_two_level_contract(
    monkeypatch, tmp_path, mapping
):
    actor = make_actor(tmp_path=tmp_path, mapping=mapping)
    fake_all_gather(monkeypatch, actor)
    actor._prepare_dvac_train_step()
    actor._write_dvac_step_artifact(actor._dvac_pending_step)
    step_path = tmp_path / "runner_step_0007.pt"
    saved = torch.load(step_path, weights_only=False)
    assert saved["schema_version"] == 2 and saved["actor_rank"] == 0
    assert saved["config"] == actor._dvac_two_level_contract()
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


def test_legacy_sidecar_accepts_only_semantically_identical_linear_mapping(tmp_path):
    actor = make_actor(tmp_path=tmp_path)
    checkpoint = str(tmp_path / "checkpoint")
    actor.save_checkpoint(checkpoint, 7)
    sidecar = actor._dvac_sidecar_path(checkpoint)
    payload = json.loads(sidecar.read_text())
    assert "mapping" not in payload["two_level_config"]
    # Explicit linear and irrelevant temperatures preserve old exact behavior.
    actor.dvac_train_cfg.update(
        mapping="linear_centered", temperature_local=0.25, temperature_chunk=9.0
    )
    actor.load_checkpoint(checkpoint)
    payload["two_level_config"].update(
        mapping="linear_centered", temperature_local=3.0, temperature_chunk=0.1
    )
    sidecar.write_text(json.dumps(payload))
    actor.load_checkpoint(checkpoint)
    actor.dvac_train_cfg["mapping"] = "exp_mean"
    with pytest.raises(ValueError, match="two-level resume configuration mismatch"):
        actor.load_checkpoint(checkpoint)


@pytest.mark.parametrize("field", ["temperature_local", "temperature_chunk", "mapping"])
def test_exp_sidecar_rejects_changed_effective_distribution(tmp_path, field):
    actor = make_actor(tmp_path=tmp_path, mapping="exp_mean")
    checkpoint = str(tmp_path / "checkpoint")
    actor.save_checkpoint(checkpoint, 7)
    actor.load_checkpoint(checkpoint)
    actor.dvac_train_cfg[field] = "linear_centered" if field == "mapping" else 0.25
    with pytest.raises(ValueError, match="two-level resume configuration mismatch"):
        actor.load_checkpoint(checkpoint)


def _gloo_actor_worker(rank, world_size, directory, mapping):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=(Path(directory) / "gloo_init").as_uri(),
        timeout=timedelta(seconds=45),
    )
    try:
        actor = make_actor(rank=rank, mapping=mapping)
        actor._prepare_dvac_train_step()
        weights = actor.rollout_batch["forward_inputs"]["dvac_weights"]
        torch.testing.assert_close(
            weights, expected_weights(mapping=mapping)[:, rank * 4 : (rank + 1) * 4]
        )
        torch.save(weights, Path(directory) / f"rank{rank}.pt")
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="CPU gloo unavailable",
)
@pytest.mark.parametrize("mapping", ["linear_centered", "exp_mean"])
def test_real_two_rank_gloo_prepare_matches_full_rollout(tmp_path, mapping):
    torch.multiprocessing.spawn(
        _gloo_actor_worker, args=(2, str(tmp_path), mapping), nprocs=2, join=True
    )
    gathered = torch.cat(
        [
            torch.load(tmp_path / f"rank{rank}.pt", weights_only=True)
            for rank in range(2)
        ],
        dim=1,
    )
    torch.testing.assert_close(gathered, expected_weights(mapping=mapping))


def make_controlled_actor(rank=0, scope="positive", probability=0.5, tmp_path=None):
    """Use distinct layer endpoints to expose schedule wiring mistakes."""
    actor = make_actor(rank=rank, scope=scope, tmp_path=tmp_path)
    actor.version = 9  # Zero-based runner step 9 trains the displayed R10.
    actor.dvac_train_cfg.update(
        alpha_local=1.0,
        alpha_chunk=1.0,
        chunk_dropout={"enabled": True, "probability": probability, "seed": 917},
        alpha_schedule={
            "enabled": True,
            "local": {
                "enabled": True,
                "start_step": 1,
                "end_step": 10,
                "end_alpha": 0.2,
            },
            "chunk": {
                "enabled": True,
                "start_step": 1,
                "end_step": 19,
                "end_alpha": 0.6,
            },
        },
    )
    return actor


@pytest.mark.parametrize("rank", [0, 1])
def test_controlled_prepare_freezes_chunk_gate_and_preserves_scope(monkeypatch, rank):
    actor = make_controlled_actor(rank=rank)
    fake_all_gather(monkeypatch, actor)
    rng_before = torch.random.get_rng_state().clone()
    actor._prepare_dvac_train_step()
    assert torch.equal(torch.random.get_rng_state(), rng_before)

    pending = actor._dvac_pending_step
    expected = expected_weights(scope="positive", alpha_local=0.2, alpha_chunk=0.8)[
        :, rank * 4 : (rank + 1) * 4
    ]
    torch.testing.assert_close(pending["pre_dropout_weights"], expected)
    dropped = pending["dropout_mask"]
    assert dropped.shape == (3, 4, 1) and dropped.dtype == torch.bool
    assert not (dropped & ~pending["eligible_mask"]).any()
    weights = actor.rollout_batch["forward_inputs"]["dvac_weights"]
    torch.testing.assert_close(weights, torch.where(dropped, 1.0, expected))
    outside = ~pending["eligible_mask"].expand_as(weights)
    assert torch.equal(weights[outside], torch.ones_like(weights[outside]))
    assert pending["metrics"]["actor/dvac_alpha_local"] == pytest.approx(0.2)
    assert pending["metrics"]["actor/dvac_alpha_chunk"] == pytest.approx(0.8)
    assert pending["metrics"]["actor/dvac_dropout_probability"] == 0.5
    # The gate travels with the weights and is not redrawn by optimizer reuse.
    with pytest.raises(ValueError, match="Missing rollout DVAC signal"):
        actor._prepare_dvac_train_step()


@pytest.mark.parametrize("scope", ["both", "positive"])
def test_controlled_prepare_full_dropout_restores_original_advantages(
    monkeypatch, scope
):
    actor = make_controlled_actor(scope=scope, probability=1.0)
    advantages = actor.rollout_batch["advantages"].detach().clone()
    fake_all_gather(monkeypatch, actor)
    actor._prepare_dvac_train_step()
    pending = actor._dvac_pending_step
    weights = actor.rollout_batch["forward_inputs"]["dvac_weights"]
    assert torch.equal(weights, torch.ones_like(weights))
    assert torch.equal(pending["dropout_mask"], pending["eligible_mask"])
    assert pending["metrics"]["actor/dvac_dropout_fraction"] == 1.0
    torch.testing.assert_close(actor.rollout_batch["advantages"], advantages)
    # The existing all-ones loss tests additionally compare actual native PPO
    # loss, metrics and gradients against Clean, including both clipping modes.


@pytest.mark.parametrize(
    "runner_step, local_alpha, chunk_alpha",
    [(0, 1.0, 1.0), (9, 0.2, 0.8), (18, 0.2, 0.6), (19, 0.2, 0.6)],
)
def test_actor_schedule_uses_absolute_round_and_independent_endpoints(
    monkeypatch, runner_step, local_alpha, chunk_alpha
):
    actor = make_controlled_actor(probability=0.0)
    actor.version = runner_step
    fake_all_gather(monkeypatch, actor)
    actor._prepare_dvac_train_step()
    expected = expected_weights(
        scope="positive", alpha_local=local_alpha, alpha_chunk=chunk_alpha
    )[:, :4]
    torch.testing.assert_close(
        actor.rollout_batch["forward_inputs"]["dvac_weights"], expected
    )
    metrics = actor._dvac_pending_step["metrics"]
    assert metrics["actor/dvac_alpha_local"] == pytest.approx(local_alpha)
    assert metrics["actor/dvac_alpha_chunk"] == pytest.approx(chunk_alpha)
    assert metrics["actor/dvac_dropout_chunks"] == 0


def test_controlled_sidecar_resume_locks_enabled_fields_and_absolute_schedule(
    monkeypatch, tmp_path
):
    actor = make_controlled_actor(tmp_path=tmp_path)
    checkpoint = str(tmp_path / "global_step_10" / "actor")
    actor.save_checkpoint(checkpoint, 10)
    actor.load_checkpoint(checkpoint)
    sidecar = json.loads(actor._dvac_sidecar_path(checkpoint).read_text())
    assert sidecar["runner_step"] == 10
    assert sidecar["two_level_config"] == actor._dvac_two_level_contract()
    for field, new_value in (("probability", 0.1), ("seed", 918)):
        original = actor.dvac_train_cfg["chunk_dropout"][field]
        actor.dvac_train_cfg["chunk_dropout"][field] = new_value
        with pytest.raises(ValueError, match="two-level resume configuration mismatch"):
            actor.load_checkpoint(checkpoint)
        actor.dvac_train_cfg["chunk_dropout"][field] = original
    for layer, field, new_value in (
        ("local", "end_step", 11),
        ("chunk", "end_alpha", 0.5),
    ):
        original = actor.dvac_train_cfg["alpha_schedule"][layer][field]
        actor.dvac_train_cfg["alpha_schedule"][layer][field] = new_value
        with pytest.raises(ValueError, match="two-level resume configuration mismatch"):
            actor.load_checkpoint(checkpoint)
        actor.dvac_train_cfg["alpha_schedule"][layer][field] = original

    # The runner restores CP10's absolute completed-step counter, then passes
    # version=10 before R11. A fresh object must not restart either schedule.
    resumed = make_controlled_actor(tmp_path=tmp_path)
    resumed.load_checkpoint(checkpoint)
    resumed.version = sidecar["runner_step"]
    fake_all_gather(monkeypatch, resumed)
    resumed._prepare_dvac_train_step()
    expected = expected_weights(
        scope="positive", alpha_local=0.2, alpha_chunk=1.0 - 0.4 * 10 / 18
    )[:, :4]
    torch.testing.assert_close(
        resumed._dvac_pending_step["pre_dropout_weights"], expected
    )


def test_explicitly_disabled_controls_accept_legacy_sidecar(tmp_path):
    actor = make_actor(tmp_path=tmp_path)
    checkpoint = str(tmp_path / "legacy")
    actor.save_checkpoint(checkpoint, 7)
    legacy = json.loads(actor._dvac_sidecar_path(checkpoint).read_text())
    assert "linear_controls" not in legacy["two_level_config"]
    actor.dvac_train_cfg.update(
        chunk_dropout={"enabled": False, "probability": 0.1, "seed": 42},
        alpha_schedule={
            "enabled": False,
            "local": {
                "enabled": True,
                "start_step": 1,
                "end_step": 10,
                "end_alpha": 0.2,
            },
            "chunk": {"enabled": False},
        },
    )
    assert actor._dvac_two_level_contract() == legacy["two_level_config"]
    actor.load_checkpoint(checkpoint)


def _gloo_controlled_actor_worker(rank, world_size, directory):
    torch.set_num_threads(1)
    torch.manual_seed(1300 + rank)  # Ambient rank RNG must not define the gate.
    torch.distributed.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=(Path(directory) / "gloo_init").as_uri(),
        timeout=timedelta(seconds=45),
    )
    try:
        actor = make_controlled_actor(rank=rank)
        actor._prepare_dvac_train_step()
        pending = actor._dvac_pending_step
        torch.save(
            {
                key: pending[key]
                for key in ("weights", "dropout_mask", "pre_dropout_weights")
            },
            Path(directory) / f"controlled_rank{rank}.pt",
        )
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="CPU gloo unavailable",
)
def test_real_two_rank_gloo_controls_match_complete_rollout(tmp_path):
    torch.multiprocessing.spawn(
        _gloo_controlled_actor_worker, args=(2, str(tmp_path)), nprocs=2, join=True
    )
    parts = [
        torch.load(tmp_path / f"controlled_rank{rank}.pt", weights_only=True)
        for rank in range(2)
    ]
    expected_pre = expected_weights(scope="positive", alpha_local=0.2, alpha_chunk=0.8)
    packets = gather_packets(sample_shards())
    eligible = torch.cat(packets[2], 1) & (torch.cat(packets[3], 1) > 0)
    actor = make_controlled_actor()
    expected, mask = apply_chunk_dropout(
        expected_pre,
        eligible,
        actor._dvac_two_level_contract()["linear_controls"],
        runner_step=9,
    )
    assert mask.any() and (eligible & ~mask).any()
    torch.testing.assert_close(
        torch.cat([part["weights"] for part in parts], 1), expected
    )
    assert torch.equal(torch.cat([part["dropout_mask"] for part in parts], 1), mask)
    torch.testing.assert_close(
        torch.cat([part["pre_dropout_weights"] for part in parts], 1), expected_pre
    )
