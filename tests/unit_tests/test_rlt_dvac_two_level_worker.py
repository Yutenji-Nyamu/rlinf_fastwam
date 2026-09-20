# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""CPU integration checks for full-batch RLT weighting and strict resume."""

import inspect
import random
from contextlib import nullcontext
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.nested_dict_process import split_dict_to_chunk
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import (
    RLTACFSDPPolicy,
    RLTACLossMixin,
)
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


class _TinyActorCritic(torch.nn.Module):
    """Deterministic CPU stand-in; real worker losses remain under test."""

    def __init__(self):
        super().__init__()
        self.actor_actions = torch.nn.Parameter(torch.linspace(0.1, 1.0, 20))
        self.critic_bias = torch.nn.Parameter(torch.tensor(0.2))

    def forward(self, *, forward_type, obs, actions=None, **_kwargs):
        if forward_type == ForwardType.SAC:
            pi = self.actor_actions.expand(obs["ref_chunk"].shape[0], -1)
            return pi, pi.new_zeros(pi.shape[0], 1), None
        if forward_type == ForwardType.SAC_Q:
            q = actions.mean(dim=-1, keepdim=True) * self.critic_bias
            return torch.cat((q, q - 0.1), dim=-1)
        raise AssertionError(f"Unexpected forward type: {forward_type}")

    def clip_grad_norm_(self, max_norm):
        return torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)


@pytest.fixture
def worker_factory(monkeypatch):
    # Avoid Ray, device setup and model loading; execute the actual RLT init.
    def base_init(worker, cfg):
        worker.cfg = cfg
        worker._rank = 0
        worker._world_size = 1
        worker.update_step = 0

    monkeypatch.setattr(EmbodiedSACFSDPPolicy, "__init__", base_init)

    def make(*, mapping="two_level_batch", mode="apply", **overrides):
        dvac = {
            "mode": mode,
            "application": "success_episode_bc",
            "success_target": "reference",
            "selected_l": 3,
            "applied_horizon": 10,
            "strength": 1.5,
            "success_scale": 1.0,
        }
        if mapping is not None:
            dvac.update(
                mapping=mapping,
                outer_scope="successful_batch",
                alpha_local=1.0,
                alpha_chunk=1.0,
                log_eps=1e-12,
                minmax_eps=1e-6,
            )
        dvac.update(overrides)
        cfg = OmegaConf.create(
            {
                "actor": {
                    "global_batch_size": 4,
                    "micro_batch_size": 2,
                    "model": {"num_action_chunks": 10, "action_dim": 2},
                    "optim": {"clip_grad": 10.0},
                    "critic_optim": {"clip_grad": 10.0},
                },
                "algorithm": {
                    "rlt_dvac": dvac,
                    "bc_weight": 2.5,
                    "q_weight": 0.45,
                    "reference_dropout_prob": 0.0,
                    "rlt_resume": {
                        "enable": True,
                        "contract": {"stage1_manifest_id": "current-ar"},
                    },
                },
                "env": {"train": {"auto_reset": False}},
            }
        )
        worker = RLTACFSDPPolicy(cfg)
        worker.model = _TinyActorCritic()
        return worker

    return make


def _batch_and_expected_weights():
    # Two success rows straddle the microbatch boundary. Their mean log-V is
    # 0 and 2, so their outer factors must be .5 and 1.5 over the full batch.
    profile = torch.tensor([-1.0, 1.0] * 5)
    log_v = torch.stack((profile, profile - 20, profile + 2, profile + 20))
    teacher_v = torch.full((4, 3, 50), 1e15)
    teacher_v[:, 1, :10] = log_v.exp()
    reference = torch.linspace(-0.8, 0.9, 80).reshape(4, 10, 2)
    batch = {
        "curr_obs": {
            "ref_chunk": reference,
            "teacher_dvac_v": teacher_v,
            "episode_success": torch.tensor([[True], [False], [True], [False]]),
        },
        "actions": torch.full((4, 20), -9.0),
    }
    expected = torch.ones(4, 10)
    expected[0] = torch.tensor([0.25, 0.75] * 5)
    expected[2] = torch.tensor([0.75, 2.25] * 5)
    return batch, expected


def _actor_loss(worker, batch):
    return inspect.unwrap(RLTACLossMixin.forward_actor)(worker, batch)


def test_full_batch_preparation_selects_l3_c10_without_rng_or_replay_mutation(
    worker_factory,
):
    worker = worker_factory()
    batch, expected = _batch_and_expected_weights()
    old_v = batch["curr_obs"]["teacher_dvac_v"].clone()
    torch_rng = torch.get_rng_state().clone()
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()

    prepared, metrics = worker._prepare_global_batch(batch, train_actor=True)

    assert prepared is not batch
    assert prepared["curr_obs"] is batch["curr_obs"]
    assert "rlt_dvac_new_weights" not in batch
    torch.testing.assert_close(batch["curr_obs"]["teacher_dvac_v"], old_v)
    weights = prepared["rlt_dvac_new_weights"]
    torch.testing.assert_close(weights, expected)
    assert not weights.requires_grad
    assert metrics["rlt_dvac_new/success_count"] == 2.0
    assert metrics["rlt_dvac/enabled"] == 1.0
    assert worker.rlt_dvac_stats is None
    assert "rlt_dvac/baseline_frozen" not in worker._rlt_dvac_baseline_metrics()
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert random.getstate() == python_rng
    now_numpy = np.random.get_state()
    assert now_numpy[0] == numpy_rng[0]
    np.testing.assert_array_equal(now_numpy[1], numpy_rng[1])
    assert now_numpy[2:] == numpy_rng[2:]


def test_actor_consumes_precomputed_weights_and_preserves_microbatch_gradient(
    worker_factory,
):
    batch, expected = _batch_and_expected_weights()
    results = []
    for split_count in (1, 2, 4):
        worker = worker_factory()
        prepared, _ = worker._prepare_global_batch(batch, train_actor=True)
        # The forward pass must consume cached batch weights, not re-estimate
        # the outer domain independently in each microbatch.
        prepared["curr_obs"] = dict(prepared["curr_obs"])
        del prepared["curr_obs"]["teacher_dvac_v"]
        total_loss = 0.0
        for microbatch in split_dict_to_chunk(prepared, split_count):
            loss, _, metrics = _actor_loss(worker, microbatch)
            (loss / split_count).backward()
            total_loss += loss.detach() / split_count
            assert metrics["rlt_dvac/success_bc_applied"] == 1.0
        results.append((total_loss, worker.model.actor_actions.grad.clone()))

        pi = worker.model.actor_actions.detach().expand(4, -1)
        error = (pi.reshape(4, 10, 2) - batch["curr_obs"]["ref_chunk"]).square()
        expected_bc = (expected * error.mean(dim=-1)).mean()
        expected_q = (pi.mean(dim=-1) * worker.model.critic_bias.detach()).mean()
        torch.testing.assert_close(total_loss, 2.5 * expected_bc - 0.45 * expected_q)
    for loss, grad in results[1:]:
        torch.testing.assert_close(loss, results[0][0])
        torch.testing.assert_close(grad, results[0][1])


@pytest.mark.parametrize(
    ("mapping", "mode", "train_actor"),
    [(None, "off", True), (None, "apply", True), ("two_level_batch", "apply", False)],
)
def test_hook_preserves_off_pure_and_critic_only_batches(
    worker_factory, mapping, mode, train_actor
):
    worker = worker_factory(mapping=mapping, mode=mode)
    batch = {"actions": torch.zeros(4, 20)}  # No DVAC fields are needed here.
    prepared, metrics = worker._prepare_global_batch(batch, train_actor=train_actor)
    assert prepared is batch
    assert metrics == {}
    default_prepared, default_metrics = EmbodiedSACFSDPPolicy._prepare_global_batch(
        worker, batch, train_actor=train_actor
    )
    assert default_prepared is batch
    assert default_metrics == {}


@pytest.mark.parametrize("mode", ["observe", "apply"])
def test_new_forward_rejects_missing_full_batch_precomputation(worker_factory, mode):
    worker = worker_factory(mode=mode)
    batch, _ = _batch_and_expected_weights()
    with pytest.raises(ValueError, match="full batch"):
        _actor_loss(worker, batch)


def test_new_rejects_partial_batch_and_multiple_actor_ranks(worker_factory):
    worker = worker_factory()
    batch, _ = _batch_and_expected_weights()
    with pytest.raises(ValueError, match="complete actor batch"):
        worker._prepare_global_batch(split_dict_to_chunk(batch, 2)[0], train_actor=True)
    worker._world_size = 2
    with pytest.raises(ValueError, match="one actor rank"):
        worker._prepare_global_batch(batch, train_actor=True)


@pytest.mark.parametrize(
    ("update_step", "train_actor", "expected_actor"),
    [(0, True, True), (1, True, False), (0, False, False)],
)
def test_real_update_loop_prepares_once_before_split_and_only_for_actor_slots(
    worker_factory, update_step, train_actor, expected_actor
):
    worker = worker_factory()
    batch, _ = _batch_and_expected_weights()
    worker.update_step = update_step
    worker.critic_actor_ratio = 2
    worker.gradient_accumulation = 2
    worker.device = torch.device("cpu")
    worker.enable_drq = False
    worker.buffer_dataloader_iter = iter([batch])
    worker.worker_timer = lambda *_args, **_kwargs: nullcontext()
    worker.qf_optimizer = torch.optim.SGD([worker.model.critic_bias], lr=0.0)
    worker.optimizer = torch.optim.SGD([worker.model.actor_actions], lr=0.0)
    worker.qf_lr_scheduler = SimpleNamespace(step=lambda: None)
    worker.lr_scheduler = SimpleNamespace(step=lambda: None)
    worker.alpha_optimizer = None
    worker.entropy_temp = SimpleNamespace(alpha=0.0)
    worker.target_model_initialized = False
    calls = []

    def prepare(self, full_batch, *, train_actor):
        calls.append(("prepare", full_batch["actions"].shape[0], train_actor))
        return RLTACLossMixin._prepare_global_batch(
            self, full_batch, train_actor=train_actor
        )

    def critic(self, microbatch):
        calls.append(("critic", microbatch["actions"].shape[0]))
        return self.model.critic_bias.square(), {}

    def actor(self, microbatch):
        calls.append(("actor", microbatch["actions"].shape[0]))
        return _actor_loss(self, microbatch)

    worker._prepare_global_batch = MethodType(prepare, worker)
    worker.forward_critic = MethodType(critic, worker)
    worker.forward_actor = MethodType(actor, worker)
    metrics = inspect.unwrap(EmbodiedSACFSDPPolicy.update_one_epoch)(
        worker, train_actor=train_actor
    )
    assert calls[0] == ("prepare", 4, expected_actor)
    assert sum(call[0] == "prepare" for call in calls) == 1
    assert [call for call in calls if call[0] == "critic"] == [("critic", 2)] * 2
    assert [call for call in calls if call[0] == "actor"] == [("actor", 2)] * (
        2 if expected_actor else 0
    )
    if expected_actor:
        assert metrics["actor/rlt_dvac_new/success_count"] == 2.0
    else:
        assert not any(key.startswith("actor/") for key in metrics)
    assert "rlt_dvac_new_weights" not in batch


def test_observe_mapping_computes_diagnostics_but_keeps_clean_actor_loss(worker_factory):
    batch, _ = _batch_and_expected_weights()
    observed = worker_factory(mode="observe")
    clean = worker_factory(mapping=None, mode="off")
    prepared, metrics = observed._prepare_global_batch(batch, train_actor=True)
    observed_loss, _, observed_metrics = _actor_loss(observed, prepared)
    clean_loss, _, _ = _actor_loss(clean, batch)
    torch.testing.assert_close(observed_loss, clean_loss)
    assert metrics["rlt_dvac_new/success_count"] == 2.0
    assert metrics["rlt_dvac/mode_apply"] == 0.0
    assert observed_metrics["rlt_dvac/success_bc_applied"] == 0.0


def test_new_and_pure_resume_remain_separate_and_new_has_no_frozen_baseline(
    worker_factory,
):
    new = worker_factory()
    pure = worker_factory(mapping=None)
    pure.rlt_dvac_stats.freeze_from_statistics(2, 3.0, 5.0)
    new_state = new._rlt_state_payload(runner_step=4)
    pure_state = pure._rlt_state_payload(runner_step=4)
    assert new_state["rlt_dvac_baseline"] is None
    assert isinstance(pure_state["rlt_dvac_baseline"], dict)
    new._restore_rlt_state(new_state)
    pure._restore_rlt_state(pure_state)
    assert new.rlt_dvac_stats is None
    for destination, state in ((new, pure_state), (pure, new_state)):
        with pytest.raises(ValueError, match="rlt_resume_contract"):
            destination._validate_rlt_state(state)

    contaminated_new = dict(new_state, rlt_dvac_baseline=pure_state["rlt_dvac_baseline"])
    with pytest.raises(ValueError, match="baseline"):
        new._validate_rlt_state(contaminated_new)
    incomplete_pure = dict(pure_state, rlt_dvac_baseline=None)
    with pytest.raises(ValueError, match="baseline"):
        pure._validate_rlt_state(incomplete_pure)
    changed = worker_factory(alpha_chunk=0.5)
    with pytest.raises(ValueError, match="rlt_resume_contract"):
        changed._validate_rlt_state(new_state)
