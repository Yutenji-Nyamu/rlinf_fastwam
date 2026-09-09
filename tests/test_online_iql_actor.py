# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Exercise production actor methods without constructing a VLA/FSDP cluster."""

import ast
import copy
from contextlib import nullcontext
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rlinf.data.online_bc import SuccessReplay, masked_fm_loss


class CleanBase:
    def set_global_step(self, step):
        self.version = step

    def forward_actor(self, batch):
        return self.toy_parameter.square()


@pytest.fixture
def actor_module():
    source = Path(inspect.getfile(SuccessReplay)).resolve().parents[1] / "workers/actor/fsdp_online_iql_policy_worker.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = []
    for original in tree.body:
        if isinstance(original, ast.FunctionDef):
            nodes.append(copy.deepcopy(original))
        if isinstance(original, ast.ClassDef) and original.name == "EmbodiedOnlineIQLFSDPPolicy":
            node = copy.deepcopy(original)
            node.bases = [ast.Name(id="CleanBase", ctx=ast.Load())]
            for method in node.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method.decorator_list = [d for d in method.decorator_list if isinstance(d, ast.Name) and d.id in {"staticmethod", "classmethod"}]
            nodes.append(node)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    namespace = {
        "torch": torch, "Path": Path, "json": json, "math": math, "os": os,
        "hashlib": hashlib, "FORMAT_VERSION": 1, "CleanBase": CleanBase,
        "stack_pixels": lambda main, wrists: main,
    }
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace


class Pool:
    def __init__(self, n=8, seed=17):
        self.n = n
        self.rng = torch.Generator().manual_seed(seed)
        self.critic_rng = torch.Generator().manual_seed(seed + 10)
        self.archive_path = Path(f"attempt_{seed}")

    def __len__(self):
        return self.n

    def _rows(self, indices):
        count = len(indices)
        return {"forward_inputs": {
            "action": indices.float().unsqueeze(-1).repeat(1, 700) / 100,
            "observation/image": indices[:, None, None, None].to(torch.uint8),
            "observation/wrist_image": torch.zeros(count, 2, 1, 1, 3, dtype=torch.uint8),
            "observation/state": torch.zeros(count, 14),
            "action_valid_mask": torch.ones(count, 50, 14, dtype=torch.bool),
            "tokenized_prompt": torch.zeros(count, 2, dtype=torch.long),
            "tokenized_prompt_mask": torch.ones(count, 2, dtype=torch.bool),
            "iql_next_image": torch.ones(count, 1, 1, 3, dtype=torch.uint8),
            "iql_reward": torch.ones(count),
        }}

    def sample(self, num_chunks):
        return self._rows(torch.randint(self.n, (num_chunks,), generator=self.rng))

    sample_actor = sample

    def sample_critic(self, num_chunks):
        return self._rows(torch.randperm(self.n, generator=self.critic_rng)[:min(self.n, num_chunks)])

    def get_stats(self):
        return {"query_records": self.n}

    def save_checkpoint(self, target):
        Path(target).mkdir(parents=True, exist_ok=True)
        torch.save({"n": self.n, "rng": self.rng.get_state(), "critic_rng": self.critic_rng.get_state()}, Path(target) / "pool.pt")

    def load_checkpoint(self, target):
        state = torch.load(Path(target) / "pool.pt", weights_only=True)
        self.n = state["n"]
        self.rng.set_state(state["rng"])
        self.critic_rng.set_state(state["critic_rng"])


class Critic:
    def __init__(self):
        self.update_count = 0
        self.calls = []

    def advantages(self, pixels, actions, batch_size):
        self.calls.append((self.update_count, len(actions)))
        advantage = actions[:, 0, 0]
        return {"weights": 1 + advantage, "advantage": advantage}

    def state_dict(self):
        return {"update_count": self.update_count, "parameter": torch.tensor(0.25)}

    def load_state_dict(self, state):
        self.update_count = state["update_count"]


def make_actor(module, tmp_path, completed=0, successes=8):
    cls = module["EmbodiedOnlineIQLFSDPPolicy"]
    actor = cls.__new__(cls)
    actor.warmup_rounds = 10
    actor.critic_batch_size = 64
    actor.expected_episodes_per_round = 4
    actor._last_completed_round = completed
    actor._completed_before = None
    actor._round_open = False
    actor._round_received_episodes = 0
    actor._phase = "success_bc"
    actor.actor_skip_count = actor.critic_skip_count = actor.iql_slots = actor.update_step = 0
    actor._micro_unweighted = []
    actor._last_weight_metrics = {}
    actor.replay_buffer = Pool(successes)
    actor.transition_replay = Pool(128, 37)
    actor.iql = Critic()
    actor.cfg = SimpleNamespace(
        actor=SimpleNamespace(global_batch_size=1024, micro_batch_size=32),
        algorithm=SimpleNamespace(update_epoch=5),
    )
    actor._rank = 0
    actor.device = "cpu"
    actor.is_weight_offloaded = actor.is_optimizer_offloaded = False
    actor.checkpoint_format = "local_shard"
    actor.model = SimpleNamespace(
        train=lambda: None,
        prepare_dagger_sft_batch=lambda batch: {"actions": batch["action"].reshape(-1, 50, 14)},
    )
    actor.optimizer = actor.lr_scheduler = None
    actor.grad_scaler = SimpleNamespace(state_dict=lambda: {}, load_state_dict=lambda state: None)
    actor.iql_contract = {"format_version": 1, "warmup_rounds": 10}
    actor.iql_contract_hash = module["_identity_hash"](actor.iql_contract)

    def save_model(**kwargs):
        root = Path(kwargs["save_path"])
        root.mkdir(parents=True, exist_ok=True)
        torch.save({"model": torch.tensor([1.0])}, root / "model-test.pt")

    actor._strategy = SimpleNamespace(save_checkpoint=save_model, load_checkpoint=lambda **kwargs: None)
    return actor


@pytest.mark.parametrize("completed,expected", [(0, "success_bc"), (8, "success_bc"), (9, "success_bc"), (10, "iql"), (99, "iql")])
def test_phase_uses_completed_rounds_not_optimizer_count(actor_module, tmp_path, completed, expected):
    actor = make_actor(actor_module, tmp_path, completed)
    actor.update_step = 99999  # Deliberately unrelated to collection rounds.
    actor.set_global_step(completed)
    assert actor._phase == expected
    assert actor.version == completed
    assert actor._completed_before + 1 == completed + 1
    with pytest.raises(ValueError, match="Round boundary"):
        actor.set_global_step(completed)


def test_zero_warmup_and_invalid_rounds(actor_module):
    phase = actor_module["phase_for_round"]
    assert phase(0, 0) == "iql"
    for args in [(-1, 10), (True, 10), (0, -1), (0, 1.5)]:
        with pytest.raises(ValueError):
            phase(*args)


@pytest.mark.parametrize("completed,successes,expected_actor", [(9, 0, 0), (9, 8, 5), (10, 0, 5)])
def test_five_slots_keep_qv_when_success_actor_is_empty(actor_module, tmp_path, monkeypatch, completed, successes, expected_actor):
    actor = make_actor(actor_module, tmp_path, completed, successes)
    actor.set_global_step(completed)
    actor._round_received_episodes = 4
    events = []

    def update_critic():
        actor.iql.update_count += 1
        events.append("qv")
        return {"iql/q_loss": 0.2}

    def update_actor(batch):
        events.append("actor")
        if completed < 10:
            assert "sample_weights" not in batch["forward_inputs"]
        else:
            data = batch["forward_inputs"]
            torch.testing.assert_close(data["sample_weights"], data["action"][:, 0] + 1)
            assert not any(key.startswith("iql_") for key in data)
        return {"dagger/actor_loss": 0.3}

    actor._update_critic_once = update_critic
    actor.update_sampled_buffer_batch = update_actor
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    metrics = actor.run_training()
    assert actor.iql.update_count == actor.iql_slots == 5
    assert actor.update_step == expected_actor
    assert events == (["qv"] * 5 if expected_actor == 0 else ["qv", "actor"] * 5)
    assert actor._last_completed_round == completed + 1
    assert metrics["iql/phase"] == float(completed >= 10)
    assert not actor._round_open


def test_weights_cover_actual1024_and_use_one_fixed_critic(actor_module, tmp_path):
    actor = make_actor(actor_module, tmp_path, 10)
    actor.iql.update_count = 17
    batch = actor.transition_replay.sample_actor(1024)
    labels = batch["forward_inputs"]["action"].clone()
    mask = batch["forward_inputs"]["action_valid_mask"].clone()
    rng = torch.get_rng_state().clone()
    result = actor._weight_actor_batch(batch)["forward_inputs"]
    assert actor.iql.calls == [(17, 64)] * 16
    torch.testing.assert_close(result["sample_weights"], labels[:, 0] + 1)
    torch.testing.assert_close(result["action"], labels, rtol=0, atol=0)
    torch.testing.assert_close(result["action_valid_mask"], mask, rtol=0, atol=0)
    assert "iql_next_image" not in result and "iql_reward" not in result
    assert not result["sample_weights"].requires_grad
    assert torch.equal(rng, torch.get_rng_state())


def test_sampling_critic_does_not_consume_success_sampler(actor_module, tmp_path):
    actor = make_actor(actor_module, tmp_path)
    control = Pool(8)
    for _ in range(5):
        actor.transition_replay.sample_critic(64)
        actor.transition_replay.sample_actor(1024)
    actual = actor.replay_buffer.sample(1024)["forward_inputs"]["action"]
    expected = control.sample(1024)["forward_inputs"]["action"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_masked_fm_full1024_matches_micro32_gradients():
    generator = torch.Generator().manual_seed(72)
    data = torch.randn(1024, 3, 2, generator=generator)
    mask = torch.rand(1024, 3, 2, generator=generator) > 0.3
    mask[:, 0, 0] = True
    weights = torch.linspace(0, 3, 1024).requires_grad_()
    full = torch.tensor(0.2, requires_grad=True)
    micro = torch.tensor(0.2, requires_grad=True)
    expected = masked_fm_loss((data - full).square(), mask, weights)
    expected.backward()
    actual = 0
    for start in range(0, 1024, 32):
        loss = masked_fm_loss((data[start:start+32] - micro).square(), mask[start:start+32], weights[start:start+32]) / 32
        actual += loss.detach()
        loss.backward()
    torch.testing.assert_close(actual, expected.detach(), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(micro.grad, full.grad, rtol=1e-6, atol=1e-6)
    assert weights.grad is None
    torch.testing.assert_close(masked_fm_loss(data.square(), mask), masked_fm_loss(data.square(), mask, torch.ones(1024)), rtol=0, atol=0)


@pytest.mark.parametrize("completed,phase", [(9, "success_bc"), (10, "iql")])
def test_complete_round_checkpoint_restores_next_phase_and_rng(actor_module, tmp_path, completed, phase):
    actor = make_actor(actor_module, tmp_path, completed)
    actor.update_step = 37
    actor.iql.update_count = completed * 5
    actor.iql_slots = completed * 5
    actor.replay_buffer.sample(5)
    actor.transition_replay.sample_critic(4)
    actor.transition_replay.sample_actor(5)
    checkpoint = tmp_path / f"global_step_{completed}" / "actor"
    saved_rng = torch.get_rng_state().clone()
    actor.save_checkpoint(checkpoint, completed)
    expected_success = actor.replay_buffer.sample(16)["forward_inputs"]["action"]
    expected_critic = actor.transition_replay.sample_critic(16)["forward_inputs"]["action"]
    expected_actor = actor.transition_replay.sample_actor(16)["forward_inputs"]["action"]
    restored = make_actor(actor_module, tmp_path)
    restored.transition_replay.archive_path = tmp_path / "new_attempt"
    torch.rand(3)
    restored.load_checkpoint(checkpoint)
    assert torch.equal(torch.get_rng_state(), saved_rng)
    assert restored.update_step == 37
    assert restored.iql.update_count == completed * 5
    restored.set_global_step(completed)
    assert restored._phase == phase
    assert restored.transition_replay.archive_path == tmp_path / "new_attempt"
    torch.testing.assert_close(restored.replay_buffer.sample(16)["forward_inputs"]["action"], expected_success, rtol=0, atol=0)
    torch.testing.assert_close(restored.transition_replay.sample_critic(16)["forward_inputs"]["action"], expected_critic, rtol=0, atol=0)
    torch.testing.assert_close(restored.transition_replay.sample_actor(16)["forward_inputs"]["action"], expected_actor, rtol=0, atol=0)


@pytest.mark.parametrize("weight_offloaded,optimizer_offloaded", [(True, True), (True, False), (False, True), (False, False)])
def test_resume_restores_offloaded_actor_before_fsdp_load(actor_module, tmp_path, weight_offloaded, optimizer_offloaded):
    actor = make_actor(actor_module, tmp_path, 10)
    checkpoint = tmp_path / "global_step_10" / "actor"
    actor.save_checkpoint(checkpoint, 10)
    restored = make_actor(actor_module, tmp_path)
    restored.is_weight_offloaded = weight_offloaded
    restored.is_optimizer_offloaded = optimizer_offloaded
    events = []

    def load_parameters(device):
        assert device == restored.device
        events.append("parameters")

    def load_optimizer(device):
        assert device == restored.device
        events.append("optimizer")

    def load_strategy(**kwargs):
        assert not restored.is_weight_offloaded
        assert not restored.is_optimizer_offloaded
        assert kwargs["model"] is restored.model
        assert kwargs["optimizers"] == [restored.optimizer]
        events.append("fsdp_checkpoint")

    restored.load_param_and_grad = load_parameters
    restored.load_optimizer = load_optimizer
    restored._strategy.load_checkpoint = load_strategy
    restored.load_checkpoint(checkpoint)
    assert events == (["parameters"] if weight_offloaded else []) + (["optimizer"] if optimizer_offloaded else []) + ["fsdp_checkpoint"]
    assert restored._last_completed_round == 10
    assert restored._phase == "iql"


@pytest.mark.parametrize("damage", ["missing_commit", "member_missing", "member_size", "contract", "round"])
def test_invalid_checkpoint_rejected_before_loading_model(actor_module, tmp_path, damage):
    actor = make_actor(actor_module, tmp_path, 10)
    checkpoint = tmp_path / "global_step_10" / "actor"
    actor.save_checkpoint(checkpoint, 10)
    marker = checkpoint / "online_iql/rank_0/COMPLETED.json"
    if damage == "missing_commit":
        marker.unlink()
    elif damage == "member_missing":
        (checkpoint / "model-test.pt").unlink()
    elif damage == "member_size":
        (checkpoint / "model-test.pt").write_bytes(b"truncated")
    else:
        receipt = json.loads(marker.read_text())
        receipt["contract_hash" if damage == "contract" else "completed_rounds"] = "bad" if damage == "contract" else 9
        marker.write_text(json.dumps(receipt))
    restored = make_actor(actor_module, tmp_path)
    restored._ensure_actor_loaded = lambda: pytest.fail("actor moved before contract validation")
    restored._strategy.load_checkpoint = lambda **kwargs: pytest.fail("model loaded before contract validation")
    with pytest.raises(ValueError):
        restored.load_checkpoint(checkpoint)


def test_in_progress_round_and_duplicate_commit_cannot_be_saved(actor_module, tmp_path):
    actor = make_actor(actor_module, tmp_path, 9)
    actor.set_global_step(9)
    with pytest.raises(ValueError, match="finished"):
        actor.save_checkpoint(tmp_path / "partial", 10)
    actor._round_open = False
    actor._last_completed_round = 10
    actor.save_checkpoint(tmp_path / "complete", 10)
    with pytest.raises(FileExistsError):
        actor.save_checkpoint(tmp_path / "complete", 10)


def test_nonfinite_is_error_not_warmup_extension(actor_module, tmp_path):
    actor = make_actor(actor_module, tmp_path, 10)
    actor.set_global_step(10)
    with pytest.raises(FloatingPointError):
        actor.validate_replay_gradients(torch.tensor(float("nan")))
    assert actor._phase == "iql" and actor.warmup_rounds == 10
