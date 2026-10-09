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


"""Production actor, replay and Norm contracts without allocating a GPU."""

import copy
import inspect
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.rlt.norm_signal import capture_expert_norm
from rlinf.algorithms.rlt.q_chunk_weighting import (
    NORM_SIGNAL_SPEC,
    selected_q_signal,
    weighted_q_mean,
)
from rlinf.algorithms.rlt.transition import extract_rlt_obs_from_forward_inputs
from rlinf.algorithms.ugrow_signal import UGROW_SIGNAL_SPEC
from rlinf.utils.nested_dict_process import split_dict_to_chunk
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import (
    RLTACFSDPPolicy,
    RLTACLossMixin,
)
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actions = torch.nn.Parameter(torch.linspace(0.1, 1, 140))

    def forward(self, *, forward_type, obs, actions=None, **kwargs):
        if actions is None:
            pi = self.actions.expand(obs["ref_chunk"].shape[0], -1)
            return pi, pi.new_zeros(pi.shape[0], 1), None
        q = (actions * obs["q_slope"]).mean(-1, keepdim=True)
        return torch.cat((q, q - 0.1), -1)


@pytest.fixture
def make(monkeypatch):
    def base_init(self, cfg):
        self.cfg, self._world_size, self._rank = cfg, 1, 0
        self.update_step = self.version = 0

    monkeypatch.setattr(EmbodiedSACFSDPPolicy, "__init__", base_init)

    def create(source="ugrow_10_5", alpha=1.0, **q_options):
        spec = UGROW_SIGNAL_SPEC if source == "ugrow_10_5" else NORM_SIGNAL_SPEC
        cfg = OmegaConf.create(
            {
                "actor": {
                    "global_batch_size": 4,
                    "micro_batch_size": 2,
                    "model": {"num_action_chunks": 10, "action_dim": 14},
                },
                "rollout": {
                    "rlt_feature_model": {
                        "openpi": {
                            "rlt_ugrow_enabled": source == "ugrow_10_5",
                            "rlt_norm_enabled": source == "norm_tail5_layers3",
                            "rlt_dvac_mode": "off",
                            "num_steps": 10,
                            "action_env_dim": 14,
                        }
                    }
                },
                "algorithm": {
                    "rlt_dvac": {"mode": "off"},
                    "q_weight": 0.45,
                    "bc_weight": 2.5,
                    "reference_dropout_prob": 0.0,
                    "rlt_q_weighting": {
                        "enabled": True,
                        "signal": source,
                        "signal_spec": copy.deepcopy(spec),
                        "alpha": alpha,
                        "temperature": 2.5,
                        "log_eps": 1e-12,
                        "minmax_eps": 1e-6,
                    },
                    "rlt_resume": {
                        "enable": True,
                        "contract": {"stage1": "same-original"},
                    },
                },
                "env": {"train": {"auto_reset": False}},
            }
        )
        cfg.algorithm.rlt_q_weighting.update(q_options)
        worker = RLTACFSDPPolicy(cfg)
        worker.model = TinyModel()
        return worker

    return create


def batch(source):
    signal = torch.tensor([0.02, 0.1, 0.2, 0.5])[:, None].expand(4, 50).clone()
    key = "teacher_ugrow_u" if source == "ugrow_10_5" else "teacher_norm"
    return {
        "curr_obs": {
            "ref_chunk": torch.linspace(-0.6, 0.9, 560).reshape(4, 10, 14),
            key: signal,
            "q_slope": torch.tensor([1.0, 3.0, 7.0, 11.0])[:, None],
            "episode_success": torch.tensor([True, False, True, False])[:, None],
        },
        "actions": torch.zeros(4, 140),
    }


def loss(w, b):
    return inspect.unwrap(RLTACLossMixin.forward_actor)(w, b)


@pytest.mark.parametrize("source", ["ugrow_10_5", "norm_tail5_layers3"])
def test_production_q_gradient_all_rows_and_microbatch(make, source):
    w = make(source)
    b = batch(source)
    prepared, metrics = w._prepare_global_batch(b, train_actor=True)
    weights = prepared["rlt_q_weights"]
    assert weights.shape == (4, 1) and not weights.requires_grad
    assert (
        weights.mean() == pytest.approx(1)
        and weights[3] > weights[2] > weights[1] > weights[0]
    )
    full, _, actual = loss(w, prepared)
    full.backward()
    grad = w.model.actions.grad.clone()
    w.model.zero_grad()
    for micro in split_dict_to_chunk(prepared, 2):
        (loss(w, micro)[0] / 2).backward()
    torch.testing.assert_close(w.model.actions.grad, grad)
    # Oracle uses all rows, independent of BC success flags; q slopes differ by state.
    actions = w.model.actions[None].expand(4, -1)
    oracle = (
        -0.45
        * (weights * (actions * b["curr_obs"]["q_slope"]).mean(-1, keepdim=True)).mean()
    )
    oracle += (
        2.5 * (actions.reshape(4, 10, 14) - b["curr_obs"]["ref_chunk"]).square().mean()
    )
    torch.testing.assert_close(full, oracle)
    assert actual["rlt_dvac/enabled"] == 0 and metrics["rlt_q/weight_std"] > 0
    untouched, stats = w._prepare_global_batch(b, train_actor=False)
    assert untouched is b and stats == {}
    b["curr_obs"]["episode_success"].logical_not_()
    again, _ = w._prepare_global_batch(b, train_actor=True)
    assert torch.equal(weights, again["rlt_q_weights"])


def test_units_restore_clean_loss_gradient_and_resume_distinguishes_signals(make):
    w = make(alpha=0)
    b = batch("ugrow_10_5")
    p, _ = w._prepare_global_batch(b, train_actor=True)
    enabled = loss(w, p)[0]
    enabled.backward()
    grad = w.model.actions.grad.clone()
    w.model.zero_grad()
    contract = w._rlt_contract()
    state = w._rlt_state_payload(runner_step=2)
    w._validate_rlt_state(state)
    with pytest.raises(ValueError, match="rlt_resume_contract"):
        make("norm_tail5_layers3")._validate_rlt_state(state)
    w.rlt_q_cfg = None
    clean = loss(w, b)[0]
    clean.backward()
    assert torch.equal(enabled, clean) and torch.equal(grad, w.model.actions.grad)
    assert contract != w._rlt_contract()
    assert make()._rlt_contract() != make("norm_tail5_layers3")._rlt_contract()


def test_range_failures_shape_and_tail_are_not_silent(make):
    w = make()
    b = batch("ugrow_10_5")
    p, _ = w._prepare_global_batch(b, train_actor=True)
    b["curr_obs"]["teacher_ugrow_u"][:, 10:] = 1e15
    q, _ = w._prepare_global_batch(b, train_actor=True)
    assert torch.equal(p["rlt_q_weights"], q["rlt_q_weights"])
    b["curr_obs"]["teacher_ugrow_u"].zero_()
    q, _ = w._prepare_global_batch(b, train_actor=True)
    assert torch.equal(q["rlt_q_weights"], torch.ones(4, 1))
    b["curr_obs"]["teacher_ugrow_u"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        w._prepare_global_batch(b, train_actor=True)
    with pytest.raises(ValueError):
        weighted_q_mean(torch.ones(4, 1), torch.ones(4))


def test_norm_transition_preserves_raw_readout():
    x = {
        "z_rl": torch.zeros(4, 8),
        "proprio": torch.zeros(4, 14),
        "ref_chunk": torch.zeros(4, 10, 14),
        "teacher_norm": torch.arange(200).float().reshape(4, 50),
    }
    saved = extract_rlt_obs_from_forward_inputs(x)
    torch.testing.assert_close(saved["teacher_norm"], x["teacher_norm"])
    assert selected_q_signal(saved, {"signal": "norm_tail5_layers3"}).shape == (4, 10)


def test_norm_reads_residual_before_final_norm_without_action_or_rng_change():
    layers = torch.nn.ModuleList([torch.nn.Linear(4, 4, bias=False) for _ in range(4)])
    model = SimpleNamespace(
        paligemma_with_expert=SimpleNamespace(
            gemma_expert=SimpleNamespace(model=SimpleNamespace(layers=layers))
        )
    )
    x = torch.randn(2, 51, 4)

    def run():
        hidden = x
        refs = []
        for layer in layers:
            hidden = layer(hidden)
            refs.append(hidden[:, -50:].detach().norm(dim=-1))
        return torch.nn.functional.layer_norm(hidden, (4,)), refs

    before = torch.get_rng_state()
    plain, expected = run()
    with capture_expert_norm(model, 50) as observed:
        actual, _ = run()
    assert torch.equal(plain, actual) and torch.equal(before, torch.get_rng_state())
    torch.testing.assert_close(torch.stack(observed), torch.stack(expected[-3:]))
    assert all(not layer._forward_hooks for layer in layers)
    with pytest.raises(ValueError):
        with capture_expert_norm(model, 50):
            pass
    assert all(not layer._forward_hooks for layer in layers)


@pytest.mark.parametrize("source", ["ugrow_10_5", "norm_tail5_layers3"])
@pytest.mark.parametrize("mode", ["drop", "anneal", "both"])
def test_q_controls_full_batch_rng_and_microbatch(make, source, mode):
    options = q_controls(mode)
    w = make(source, **options)
    w.version, w.update_step = 1, 101
    b = batch(source)
    rng = torch.get_rng_state().clone()
    p, stats = w._prepare_global_batch(b, train_actor=True)
    assert torch.equal(rng, torch.get_rng_state())
    assert stats["rlt_q/alpha_chunk"] == (0.5 if mode != "drop" else 1.0)
    assert stats["rlt_q/runner_round"] == 2
    assert stats["rlt_q/update_step"] == 101
    full = loss(w, p)[0]
    full.backward()
    expected = w.model.actions.grad.clone()
    w.model.zero_grad()
    for micro in split_dict_to_chunk(p, 2):
        (loss(w, micro)[0] / 2).backward()
    torch.testing.assert_close(w.model.actions.grad, expected)
    # Q allocation (including dropout) must not inherit BC's success-only mask.
    b["curr_obs"]["episode_success"].logical_not_()
    again, _ = w._prepare_global_batch(b, train_actor=True)
    assert torch.equal(p["rlt_q_weights"], again["rlt_q_weights"])


def q_controls(mode):
    return {
        "chunk_dropout": {
            "enabled": mode in ("drop", "both"),
            "probability": 0.2,
            "seed": 42,
        },
        "alpha_schedule": {
            "enabled": mode in ("anneal", "both"),
            "local": {"enabled": False},
            "chunk": {"start_step": 1, "end_step": 3, "end_alpha": 0.0},
        },
    }


@pytest.mark.parametrize("endpoint", ["drop", "anneal"])
def test_q_controls_endpoints_restore_clean_loss_and_gradients(make, endpoint):
    options = q_controls(endpoint)
    options["chunk_dropout"]["probability"] = 1.0
    w = make(**options)
    w.version = 2
    b = batch("ugrow_10_5")
    p, stats = w._prepare_global_batch(b, train_actor=True)
    assert torch.equal(p["rlt_q_weights"], torch.ones(4, 1))
    weighted = loss(w, p)[0]
    weighted.backward()
    expected = w.model.actions.grad.clone()
    w.model.zero_grad()
    w.rlt_q_cfg = None
    clean = loss(w, b)[0]
    clean.backward()
    assert torch.equal(weighted, clean)
    assert torch.equal(w.model.actions.grad, expected)
    assert stats["rlt_q/dropout_fraction"] == (1.0 if endpoint == "drop" else 0.0)


def test_q_controls_resume_temperature_and_legacy_contract(make):
    assert make()._rlt_contract() == make(**q_controls("none"))._rlt_contract()
    w = make(**q_controls("both"))
    w.version, w.update_step = 1, 103
    state = w._rlt_state_payload(runner_step=w.version)
    resumed = make(**q_controls("both"))
    resumed._validate_rlt_state(state)
    resumed.version = state["saved_runner_step"]
    resumed.update_step = state["update_step"]
    b = batch("ugrow_10_5")
    actual, _ = w._prepare_global_batch(b, train_actor=True)
    restored, _ = resumed._prepare_global_batch(b, train_actor=True)
    assert torch.equal(actual["rlt_q_weights"], restored["rlt_q_weights"])
    for opts in (
        q_controls("drop"),
        q_controls("anneal"),
        {**q_controls("both"), "temperature": 3.0},
    ):
        with pytest.raises(ValueError, match="rlt_resume_contract"):
            make(**opts)._validate_rlt_state(state)
    low, _ = make(temperature=1.0)._prepare_global_batch(b, train_actor=True)
    high, _ = make(temperature=3.0)._prepare_global_batch(b, train_actor=True)
    assert low["rlt_q_weights"].std() > high["rlt_q_weights"].std()


def test_q_controls_validate_scalar_schedule_and_probability(make):
    opts = q_controls("both")
    opts["alpha_schedule"]["local"]["enabled"] = True
    with pytest.raises(ValueError, match="local.enabled=false"):
        make(**opts)
    opts = q_controls("drop")
    opts["chunk_dropout"]["probability"] = 1.1
    with pytest.raises(ValueError, match="probability"):
        make(**opts)
