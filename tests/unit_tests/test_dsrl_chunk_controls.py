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

"""DSRL chunk controls, genuine actor gradients and checkpoint compatibility."""

import copy
import inspect

import pytest
import torch
from test_dsrl_target_shadow_resume import _make_worker
from test_dsrl_u_actor import _batch, _worker

from rlinf.algorithms.dsrl_chunk_controls import dsrl_controls_contract
from rlinf.algorithms.dsrl_ugrow import (
    build_dsrl_u_weights,
    make_signal_spec,
    validate_signal_spec,
)


def controls(dropout=True, schedule=True):
    return dsrl_controls_contract(
        {
            "chunk_dropout": {"enabled": dropout, "probability": 0.2, "seed": 42},
            "alpha_schedule": {
                "enabled": schedule,
                "start_step": 1,
                "end_step": 200,
                "end_alpha": 0.0,
            },
        }
    )


@pytest.mark.parametrize(
    "dropout,schedule", [(False, False), (True, False), (False, True), (True, True)]
)
def test_controls_preserve_real_actor_global_microbatch_gradient(dropout, schedule):
    worker, batch = _worker(), _batch()
    raw = torch.tensor([0.01, 0.02, 0.2, 0.8])[:, None].expand(-1, 10)
    weights, _ = build_dsrl_u_weights(
        raw,
        torch.ones_like(raw, dtype=torch.bool),
        controls=controls(dropout, schedule),
        runner_step=99,
        update_step=7,
    )
    batch["dsrl_u_weight"] = weights
    loss, _, _ = inspect.unwrap(worker.forward_actor)(worker, batch)
    loss.backward()
    expected = ((0.7 * 2 - 3) * weights * batch["curr_obs"]["states"]).mean()
    torch.testing.assert_close(worker.model.theta.grad, expected)
    worker.model.theta.grad = None
    for start in (0, 2):
        micro = {
            "curr_obs": {"states": batch["curr_obs"]["states"][start : start + 2]},
            "dsrl_u_weight": weights[start : start + 2],
        }
        term, _, _ = inspect.unwrap(worker.forward_actor)(worker, micro)
        (term / 2).backward()
    torch.testing.assert_close(worker.model.theta.grad, expected)


def test_dropout_is_private_reproducible_and_neutralizes_without_rescaling():
    raw = torch.arange(1.0, 2561.0).reshape(256, 10)
    mask = torch.ones_like(raw, dtype=torch.bool)
    original, _ = build_dsrl_u_weights(raw, mask)
    rng = torch.random.get_rng_state()
    first, metrics = build_dsrl_u_weights(
        raw, mask, controls=controls(True, False), update_step=12
    )
    again, _ = build_dsrl_u_weights(
        raw, mask, controls=controls(True, False), update_step=12
    )
    following, _ = build_dsrl_u_weights(
        raw, mask, controls=controls(True, False), update_step=13
    )
    assert torch.equal(rng, torch.random.get_rng_state()) and torch.equal(first, again)
    assert not torch.equal(first, following)
    dropped = first == 1.0
    assert 0 < dropped.sum() < 256 and metrics["chunk_dropout_fraction"] > 0
    assert torch.equal(first[~dropped], original[~dropped])


def test_anneal_uses_runner_round_and_exact_neutral_endpoint():
    raw = torch.arange(1.0, 81.0).reshape(8, 10)
    mask = torch.ones_like(raw, dtype=torch.bool)
    original, _ = build_dsrl_u_weights(raw, mask)
    first, _ = build_dsrl_u_weights(
        raw, mask, controls=controls(False, True), runner_step=0
    )
    last, m = build_dsrl_u_weights(
        raw, mask, controls=controls(True, True), runner_step=199, update_step=100000
    )
    midpoint, mid = build_dsrl_u_weights(
        raw, mask, controls=controls(False, True), runner_step=99
    )
    assert torch.equal(first, original) and torch.equal(last, torch.ones_like(last))
    assert m["alpha_chunk_effective"] == 0
    torch.testing.assert_close(
        midpoint, 1 + mid["alpha_chunk_effective"] * (original - 1)
    )


def test_disabled_controls_keep_legacy_trainer_contract():
    worker = _worker()
    before = copy.deepcopy(worker.dsrl_u_contract)
    worker.cfg.algorithm.dsrl_u.chunk_dropout = {"enabled": False, "probability": 0.2}
    worker.cfg.algorithm.dsrl_u.alpha_schedule = {"enabled": False, "end_step": 200}
    worker._setup_dsrl_u_contract({"schema_version": 2})
    assert worker.dsrl_u_contract == before
    worker.cfg.algorithm.dsrl_u.chunk_dropout.enabled = True
    worker._setup_dsrl_u_contract({"schema_version": 2})
    assert "chunk_dropout" in worker.dsrl_u_contract["controls"]


@pytest.mark.parametrize(
    "kind,steps,side", [("ugrow_ode4_vs2", 4, 2), ("norm_residual_t4_l3", 4, 2)]
)
def test_pi0_signal_contract_is_distinct_and_strict(kind, steps, side):
    spec = make_signal_spec(
        signal_kind=kind, main_steps=steps, side_steps=side, chunk_length=20
    )
    assert validate_signal_spec(spec) == spec
    for key, value in [
        ("main_steps", 10),
        ("name", "ugrow_ode10_vs5"),
        ("chunk_length", True),
    ]:
        with pytest.raises(ValueError):
            validate_signal_spec(dict(spec, **{key: value}))


@pytest.mark.parametrize(
    "field,changed",
    [
        ("temperature", 1.0),
        ("controls", controls(False, True)),
        ("signal_spec", make_signal_spec()),
    ],
)
def test_resume_rejects_method_change_but_restores_dropout_counter(
    tmp_path, field, changed
):
    worker = _make_worker()
    worker.use_dsrl_u = True
    worker.update_step = 17
    worker.dsrl_u_contract = {
        "temperature": 2.5,
        "controls": controls(),
        "signal_spec": make_signal_spec(
            signal_kind="ugrow_ode4_vs2", main_steps=4, side_steps=2, chunk_length=20
        ),
    }
    worker._save_dsrl_trainer_state(str(tmp_path))
    resumed = _make_worker()
    resumed.use_dsrl_u = True
    resumed.target_model.load_state_dict(worker.target_model.state_dict())
    resumed.dsrl_u_contract = copy.deepcopy(worker.dsrl_u_contract)
    resumed.dsrl_u_contract[field] = changed
    with pytest.raises(ValueError, match="dsrl_u_contract"):
        resumed._load_dsrl_trainer_state(str(tmp_path))
    resumed.dsrl_u_contract = copy.deepcopy(worker.dsrl_u_contract)
    resumed._load_dsrl_trainer_state(str(tmp_path))
    assert resumed.update_step == 17


@pytest.mark.parametrize(
    "config",
    [
        {"alpha_chunk": True},
        {"chunk_dropout": {"enabled": "true"}},
        {"chunk_dropout": {"enabled": True, "probability": 1.1}},
        {"alpha_schedule": {"enabled": True, "start_step": 20, "end_step": 20}},
    ],
)
def test_invalid_controls_fail_before_collection(config):
    with pytest.raises(ValueError):
        dsrl_controls_contract(config)
