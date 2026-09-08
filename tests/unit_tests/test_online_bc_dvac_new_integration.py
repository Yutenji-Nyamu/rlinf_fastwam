# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Real actor methods with lightweight infrastructure; run on the server only.

AST loading removes import-time FSDP/OpenPI requirements, not method bodies.
The update test executes the production replay/update loop with torch autograd.
"""

import ast
import asyncio
import copy
import inspect
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.online_bc_dvac_two_level import compute_two_level_bc_weights
from rlinf.data.online_bc import SuccessEpisodeCollector, SuccessReplay, masked_fm_loss


def _source_root():
    root = Path(os.environ.get("RLINF_BC_NEW_SOURCE_ROOT", Path(inspect.getfile(SuccessReplay)).resolve().parents[2]))
    assert (root / "rlinf/workers/actor/fsdp_online_bc_policy_worker.py").is_file()
    return root


def _load_actor_class(path, name, base, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    node = copy.deepcopy(node)
    node.bases = [ast.Name(id="_TestBase", ctx=ast.Load())]
    for method in node.body:
        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
            method.decorator_list = [
                decorator for decorator in method.decorator_list
                if isinstance(decorator, ast.Name) and decorator.id in ("staticmethod", "classmethod")
            ]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    namespace["_TestBase"] = base
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def _split_batch(batch, count):
    def piece(value, index):
        if isinstance(value, dict):
            return {key: piece(item, index) for key, item in value.items()}
        return value.chunk(count, dim=0)[index]

    return [piece(batch, index) for index in range(count)]


@pytest.fixture
def actor_types():
    namespace = {
        "torch": torch,
        "np": np,
        "Path": Path,
        "SuccessReplay": SuccessReplay,
        "OmegaConf": OmegaConf,
        "ForwardType": SimpleNamespace(SFT="sft"),
        "compute_split_num": lambda send, recv: send // recv,
        "split_dict_to_chunk": _split_batch,
        "put_tensor_device": lambda batch, device: batch,
        "compute_two_level_bc_weights": compute_two_level_bc_weights,
    }
    source = _source_root() / "rlinf/workers/actor"
    dagger = _load_actor_class(
        source / "fsdp_dagger_policy_worker.py", "EmbodiedDAGGERFSDPPolicy", object, namespace
    )
    online = _load_actor_class(
        source / "fsdp_online_bc_policy_worker.py", "EmbodiedOnlineBCFSDPPolicy", dagger, namespace
    )
    return dagger, online


def _actor(actor_types, *, normalization="two_level_batch", batch_size=1024):
    actor = actor_types[1].__new__(actor_types[1])
    actor.cfg = OmegaConf.create(
        {
            "actor": {"global_batch_size": batch_size, "micro_batch_size": 32, "optim": {"clip_grad": 1e6}},
            "algorithm": {"online_bc": {"demo_weight": 0.0}},
        }
    )
    actor._rank = 0
    actor._world_size = 1
    actor.device = "cpu"
    actor.dvac_normalization = normalization
    actor.dvac_new_settings = dict(alpha_local=1.0, alpha_chunk=1.0, variance_eps=1e-12, range_eps=1e-6)
    actor.dvac = None
    actor.dvac_metrics = {}
    actor.dvac_batch_metrics = {}
    actor.dvac_debug_batches = 0
    actor.update_step = 0
    actor.worker_timer = lambda *args, **kwargs: nullcontext()
    actor.log_info = lambda *args, **kwargs: None
    actor.demo_weight = 0.0
    return actor


def _batch(size=1024):
    index = torch.arange(size) % 17
    variance = torch.stack((index.float() / 4 - 5, index.float() / 9 - 3, index.float() / 7 + 1), -1).exp()
    return {"forward_inputs": {
        "dvac_v": variance,
        "action_valid_mask": torch.ones(size, 3, 2, dtype=torch.bool),
        "action_weights": torch.full((size, 3), 999.0),
        "action": torch.arange(size * 6, dtype=torch.float32).reshape(size, 6),
        "query_idx": index,
    }}


def test_parent_and_legacy_hooks_preserve_old_weights_and_object(actor_types):
    batch = _batch()
    parent = actor_types[0].__new__(actor_types[0])
    assert parent.prepare_replay_batch(batch) is batch
    actor = _actor(actor_types, normalization="recent")
    assert actor.prepare_replay_batch(batch) is batch
    assert batch["forward_inputs"]["action_weights"].eq(999).all()
    assert not actor.dvac_batch_metrics


@pytest.mark.parametrize("normalization", ["recent", "two_level_batch"])
def test_initialization_selects_separate_state_and_preserves_legacy_default(actor_types, tmp_path, normalization):
    actor = _actor(actor_types)
    actor.cfg.actor.seed = 42
    actor.cfg.actor.fsdp_config = {"checkpoint_format": "local_shard"}
    actor.cfg.algorithm.online_bc.data_path = str(tmp_path / "pool")
    config = {"enabled": True, "tail_steps": 3, "log_eps": 1e-12}
    if normalization == "recent":
        config.update(mapping="bounded_linear", weight_min=0.0, weight_max=5.0)
    else:
        config.update(normalization="two_level_batch", alpha_local=1.0, alpha_chunk=1.0, range_eps=1e-6)
    actor.cfg.algorithm.online_bc.dvac = config
    actor.setup_dagger_components()
    assert actor.dvac_normalization == normalization
    if normalization == "recent":
        assert actor.dvac.settings["mapping"] == "bounded_linear"
        assert actor.dvac.settings["weight_max"] == 5.0 and len(actor.dvac.history) == 0
    else:
        assert actor.dvac is None
        assert actor.dvac_new_state()["normalization"] == "two_level_batch"
        assert "history" not in actor.dvac_new_state()


@pytest.mark.parametrize("damage", ["multiple_ranks", "legacy_settings", "disabled"])
def test_new_initialization_rejects_ambiguous_domains_or_legacy_settings(actor_types, tmp_path, damage):
    actor = _actor(actor_types)
    actor.cfg.actor.seed = 42
    actor.cfg.actor.fsdp_config = {"checkpoint_format": "local_shard"}
    actor.cfg.algorithm.online_bc.data_path = str(tmp_path / "pool")
    actor.cfg.algorithm.online_bc.dvac = {"enabled": True, "normalization": "two_level_batch"}
    if damage == "multiple_ranks":
        actor._world_size = 2
    elif damage == "legacy_settings":
        actor.cfg.algorithm.online_bc.dvac.window = 5
    else:
        actor.cfg.algorithm.online_bc.dvac.enabled = False
    with pytest.raises(ValueError):
        actor.setup_dagger_components()


def test_full_batch_hook_replaces_old_weights_preserves_signal_and_duplicates(actor_types):
    actor = _actor(actor_types)
    batch = _batch()
    original_v = batch["forward_inputs"]["dvac_v"].clone()
    expected, _ = compute_two_level_bc_weights(original_v, batch["forward_inputs"]["action_valid_mask"])
    result = actor.prepare_replay_batch(batch)
    weights = result["forward_inputs"]["action_weights"]
    torch.testing.assert_close(weights, expected)
    torch.testing.assert_close(result["forward_inputs"]["dvac_v"], original_v, rtol=0, atol=0)
    torch.testing.assert_close(weights[0], weights[17], rtol=0, atol=0)
    assert not weights.requires_grad and weights.max() < 4
    assert actor.dvac_batch_metrics["dvac_new/weight_mean"] == pytest.approx(1, abs=1e-6)


@pytest.mark.parametrize("damage", ["missing_v", "empty_query", "partial_batch"])
def test_new_hook_rejects_incomplete_online_supervision(actor_types, damage):
    actor = _actor(actor_types)
    batch = _batch(32 if damage == "partial_batch" else 1024)
    if damage == "missing_v":
        del batch["forward_inputs"]["dvac_v"]
    elif damage == "empty_query":
        batch["forward_inputs"]["action_valid_mask"][0].zero_()
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        actor.prepare_replay_batch(batch)


def test_actual_optimizer_loop_normalizes_1024_once_before_32_microbatches(actor_types):
    actor = _actor(actor_types)
    source_batch = _batch()
    expected, _ = compute_two_level_bc_weights(source_batch["forward_inputs"]["dvac_v"], source_batch["forward_inputs"]["action_valid_mask"])
    seen = []
    sampled = []
    hook_sizes = []
    original_hook = actor.prepare_replay_batch

    def sample(*, num_chunks):
        sampled.append(num_chunks)
        return copy.deepcopy(source_batch)

    def prepare(batch):
        hook_sizes.append(batch["forward_inputs"]["dvac_v"].shape[0])
        return original_hook(batch)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.probe = torch.nn.Parameter(torch.tensor(1.0))

        def clip_grad_norm_(self, max_norm):
            return torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)

    actor.model = Model()
    actor.replay_buffer = SimpleNamespace(sample=sample)
    actor.prepare_replay_batch = prepare
    actor.gradient_accumulation = 32
    actor.enable_drq = False
    actor.amp_context = nullcontext()
    actor.before_micro_batch = lambda *args, **kwargs: nullcontext()
    actor.grad_scaler = SimpleNamespace(scale=lambda loss: loss)
    actor.optimizer = torch.optim.SGD(actor.model.parameters(), lr=0.01)
    actor.lr_scheduler = SimpleNamespace(step=lambda: None)

    def forward(batch):
        seen.append(batch["action_weights"].clone())
        errors = actor.model.probe.square() * batch["dvac_v"].unsqueeze(-1).expand(-1, -1, 2)
        return masked_fm_loss(errors, batch["action_valid_mask"], batch["action_weights"])

    actor.forward_actor = forward
    metrics = actor.update_buffer_one_epoch()
    assert sampled == [1024] and hook_sizes == [1024]
    assert len(seen) == 32 and all(w.shape == (32, 3) for w in seen)
    torch.testing.assert_close(torch.cat(seen), expected, rtol=0, atol=0)
    assert source_batch["forward_inputs"]["action_weights"].eq(999).all()
    full_batch_error = source_batch["forward_inputs"]["dvac_v"].unsqueeze(-1).expand(-1, -1, 2)
    full_batch_loss = masked_fm_loss(full_batch_error, source_batch["forward_inputs"]["action_valid_mask"], expected)
    assert actor.model.probe.item() == pytest.approx(1 - 0.01 * 2 * full_batch_loss.item(), abs=2e-6)
    assert metrics["dvac_new/weight_mean"] == pytest.approx(1, abs=1e-6)


class _PacketChannel:
    def __init__(self, packets):
        self.packets = iter(packets)

    def get(self, **kwargs):
        packet = next(self.packets)

        async def result():
            return packet

        return SimpleNamespace(async_wait=result)


def _record():
    return {key: value[0].clone() for key, value in _batch(1)["forward_inputs"].items() if key != "action_weights"}


def test_new_receive_accepts_episode_packet_without_history_or_annotation(actor_types, tmp_path, monkeypatch):
    actor = _actor(actor_types)
    actor.stage_num = 1
    actor._component_placement = SimpleNamespace(get_world_size=lambda name: 1)
    actor.replay_buffer = SuccessReplay(42, str(tmp_path / "pool"))
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *a, **k: pytest.fail("new reception must not reduce historical moments"))
    row = _record()
    original_v = row["dvac_v"].clone()
    asyncio.run(actor.recv_rollout_trajectories(_PacketChannel([{"episodes": [[row]]}])))
    assert len(actor.replay_buffer) == 1
    saved = actor.replay_buffer.records[0]
    assert "action_weights" not in saved and "dvac_calibration_round" not in saved
    torch.testing.assert_close(saved["dvac_v"], original_v, rtol=0, atol=0)
    assert actor.dvac is None


def test_legacy_receive_still_annotates_and_reduces_old_moments(actor_types, tmp_path, monkeypatch):
    actor = _actor(actor_types, normalization="recent")
    actor.stage_num = 1
    actor._component_placement = SimpleNamespace(get_world_size=lambda name: 1)
    actor.replay_buffer = SuccessReplay(42, str(tmp_path / "pool"))
    calls = []
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda value: calls.append("reduce"))

    def annotate(episodes, moments):
        calls.append("annotate")
        episodes[0][0]["action_weights"] = torch.full((3,), 0.25)
        return {"dvac/reference_positions": moments[0].item()}

    actor.dvac = SimpleNamespace(annotate=annotate)
    packet = {"episodes": [[_record()]], "dvac_moments": torch.tensor([3., 2., 4.])}
    asyncio.run(actor.recv_rollout_trajectories(_PacketChannel([packet])))
    assert calls == ["reduce", "annotate"]
    assert actor.replay_buffer.records[0]["action_weights"].eq(0.25).all()


@pytest.mark.parametrize("bad_mode", ["missing", "old", "wrong_settings", "wrong_version"])
def test_new_checkpoint_rejects_incompatible_state_before_loading_model(actor_types, tmp_path, bad_mode):
    actor = _actor(actor_types)
    actor._strategy = SimpleNamespace(load_checkpoint=lambda **kwargs: pytest.fail("model loading must follow algorithm contract validation"))
    actor.checkpoint_format = "local_shard"
    actor.model = actor.optimizer = actor.lr_scheduler = None
    target = tmp_path / "online_bc/rank_0"
    target.mkdir(parents=True)
    state = {"normalization": "two_level_batch", "version": 1, "settings": actor.dvac_new_settings.copy()}
    if bad_mode == "old":
        torch.save({"history": []}, target / "dvac.pt")
    elif bad_mode == "wrong_settings":
        state["settings"]["alpha_chunk"] = 0.5
        torch.save(state, target / "dvac_new.pt")
    elif bad_mode == "wrong_version":
        state["version"] = -1
        torch.save(state, target / "dvac_new.pt")
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        actor.load_checkpoint(tmp_path)


def test_new_checkpoint_roundtrip_preserves_raw_variance_without_legacy_state(actor_types, tmp_path):
    actor = _actor(actor_types)
    calls = []
    actor._strategy = SimpleNamespace(
        save_checkpoint=lambda **kwargs: calls.append("save"),
        load_checkpoint=lambda **kwargs: calls.append("load"),
    )
    actor.checkpoint_format = "local_shard"
    actor.model = actor.optimizer = actor.lr_scheduler = None
    actor.is_weight_offloaded = actor.is_optimizer_offloaded = False
    actor.replay_buffer = SuccessReplay(7, str(tmp_path / "original_pool"))
    batch = _batch(17)["forward_inputs"]
    rows = [{key: value[index].clone() for key, value in batch.items() if key != "action_weights"} for index in range(17)]
    actor.replay_buffer.add_episodes([rows])
    actor.update_step = 12
    target = tmp_path / "checkpoint"
    actor.save_checkpoint(target, 3)
    expected_draw = actor.replay_buffer.sample(1024)
    saved = target / "online_bc/rank_0"
    assert (saved / "dvac_new.pt").is_file() and not (saved / "dvac.pt").exists()
    actor.replay_buffer = SuccessReplay(99, str(tmp_path / "restored_pool"))
    actor.update_step = 0
    actor.load_checkpoint(target)
    assert calls == ["save", "load"] and actor.update_step == 12
    torch.testing.assert_close(actor.replay_buffer.records[0]["dvac_v"], _record()["dvac_v"])
    assert "action_weights" not in actor.replay_buffer.records[0]
    restored_draw = actor.replay_buffer.sample(1024)
    for key in expected_draw["forward_inputs"]:
        torch.testing.assert_close(restored_draw["forward_inputs"][key], expected_draw["forward_inputs"][key], rtol=0, atol=0)


@pytest.mark.parametrize("damage", ["missing_v", "frozen_weights"])
def test_new_checkpoint_rejects_bad_replay_before_model_load(actor_types, tmp_path, damage):
    actor = _actor(actor_types)
    actor._strategy = SimpleNamespace(load_checkpoint=lambda **kwargs: pytest.fail("invalid replay must fail before model load"))
    actor.checkpoint_format = "local_shard"
    actor.model = actor.optimizer = actor.lr_scheduler = None
    actor.replay_buffer = SuccessReplay(7, str(tmp_path / "pool"))
    row = _record()
    if damage == "missing_v":
        del row["dvac_v"]
    else:
        row["action_weights"] = torch.ones(3)
    actor.replay_buffer.add_episodes([[row]])
    saved = tmp_path / "online_bc/rank_0"
    actor.replay_buffer.save_checkpoint(saved)
    torch.save(actor.dvac_new_state(), saved / "dvac_new.pt")
    with pytest.raises(ValueError):
        actor.load_checkpoint(tmp_path)


def test_new_collector_records_fixed_variance_without_moment_history():
    collector = SuccessEpisodeCollector(1, dvac_log_eps=1e-12, collect_dvac_moments=False)
    variance = torch.tensor([[0.1, 0.2, 0.3]])
    collector.append(
        {"observation/state": torch.ones(1, 2), "dvac_v": variance},
        torch.ones(1, 3, 2), [True], torch.ones(1, 3),
    )
    variance.zero_()
    row = collector.drain()[0][0]
    torch.testing.assert_close(row["dvac_v"], torch.tensor([0.1, 0.2, 0.3]))
    assert collector.drain_dvac_moments().eq(0).all()
    assert "action_weights" not in row


def test_replay_replacement_sampling_keeps_variance_command_indices_aligned(tmp_path):
    replay = SuccessReplay(7, str(tmp_path / "pool"))
    batch = _batch(17)["forward_inputs"]
    rows = [{key: value[index].clone() for key, value in batch.items()} for index in range(17)]
    replay.add_episodes([rows])
    actual = replay.sample(1024)["forward_inputs"]
    indices = actual["query_idx"]
    torch.testing.assert_close(actual["dvac_v"], batch["dvac_v"][indices], rtol=0, atol=0)
    torch.testing.assert_close(actual["action"], batch["action"][indices], rtol=0, atol=0)
    assert len(indices.unique()) < 1024
    actual["dvac_v"].zero_()
    torch.testing.assert_close(replay.records[0]["dvac_v"], batch["dvac_v"][0], rtol=0, atol=0)
