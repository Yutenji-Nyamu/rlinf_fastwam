# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Temperature allocation and the actual full-batch worker/resume wiring."""

import ast
import copy
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.rlt.dvac_two_level import build_two_level_success_weights


def _example():
    variances = torch.exp(
        torch.tensor([[0.0, 1.0], [2.0, 3.0], [-20.0, 20.0]], dtype=torch.float64)
    )
    return variances, torch.tensor([True, True, False])


def test_default_and_explicit_linear_keep_legacy_bits_and_ignore_temperature():
    v, success = _example()
    legacy, metrics = build_two_level_success_weights(v, success, log_eps=1e-30)
    assert torch.equal(
        legacy, torch.tensor([[0.25, 0.75], [0.75, 2.25], [1.0, 1.0]])
    )
    for local, chunk in ((0.5, 1.0), (2.0, 10.0)):
        explicit, actual_metrics = build_two_level_success_weights(
            v, success, log_eps=1e-30, factor_mapping="linear_centered",
            temperature_local=local, temperature_chunk=chunk,
        )
        assert torch.equal(explicit, legacy)
        assert actual_metrics == metrics


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_exp_matches_two_point_closed_form_and_preserves_success_scale(temperature):
    v, success = _example()
    weights, metrics = build_two_level_success_weights(
        v, success, log_eps=1e-30, factor_mapping="exp_mean",
        temperature_local=temperature, temperature_chunk=temperature,
        success_scale=2.5,
    )
    # Both MinMax domains are [0,1]; their factor ratio is exp(1/tau).
    high = 2.0 / (1.0 + math.exp(-1.0 / temperature))
    low = 2.0 - high
    expected = torch.tensor(
        [[2.5*low*low, 2.5*low*high], [2.5*high*low, 2.5*high*high], [1., 1.]]
    )
    torch.testing.assert_close(weights, expected)
    assert torch.equal(weights[~success], torch.ones_like(weights[~success]))
    assert metrics["rlt_dvac_new/success_applied_mean"] == pytest.approx(2.5)
    assert metrics["rlt_dvac_new/inner_mean"] == pytest.approx(1.0)
    assert metrics["rlt_dvac_new/outer_mean"] == pytest.approx(1.0)
    assert metrics["rlt_dvac_new/temperature_local"] == temperature
    assert metrics["rlt_dvac_new/temperature_chunk"] == temperature


def test_lower_temperature_concentrates_and_failures_do_not_set_domain():
    v, success = _example()
    results = []
    for tau in (0.5, 1.0, 2.0):
        w, metrics = build_two_level_success_weights(
            v, success, factor_mapping="exp_mean",
            temperature_local=tau, temperature_chunk=tau,
        )
        results.append((w, metrics["rlt_dvac_new/success_applied_ess_ratio"]))
    assert results[0][1] < results[1][1] < results[2][1]
    assert results[0][0].max() > results[1][0].max() > results[2][0].max()
    altered = v.clone(); altered[-1] = torch.tensor([0., 1e100], dtype=v.dtype)
    w, _ = build_two_level_success_weights(
        altered, success, factor_mapping="exp_mean",
        temperature_local=0.5, temperature_chunk=0.5,
    )
    assert torch.equal(w, results[0][0])


@pytest.mark.parametrize("kind", ["constant", "near_constant", "no_success", "empty"])
def test_tied_and_empty_domains_stay_neutral_and_finite(kind):
    if kind == "near_constant":
        v = torch.exp(torch.tensor([[0., 1e-8], [2e-8, 3e-8]], dtype=torch.float64))
    else:
        v = torch.ones((0, 10) if kind == "empty" else (2, 10))
    success = torch.full((v.shape[0],), kind != "no_success", dtype=torch.bool)
    w, metrics = build_two_level_success_weights(
        v, success, factor_mapping="exp_mean", temperature_local=0.5,
        temperature_chunk=0.5, success_scale=2.0,
    )
    expected = torch.ones_like(w); expected[success] = 2.0
    assert torch.equal(w, expected)
    assert all(math.isfinite(value) for value in metrics.values())


def test_local_and_outer_temperatures_are_independent_and_alpha_zero_is_identity():
    v, success = _example()
    for active in ("local", "chunk"):
        kwargs = dict(factor_mapping="exp_mean", alpha_local=0., alpha_chunk=0.)
        kwargs["alpha_" + active] = 1.0
        kwargs["temperature_" + active] = 0.5
        first, _ = build_two_level_success_weights(v, success, **kwargs)
        kwargs["temperature_" + ("chunk" if active == "local" else "local")] = 10.
        second, _ = build_two_level_success_weights(v, success, **kwargs)
        assert torch.equal(first, second)
    w, _ = build_two_level_success_weights(
        v, success, factor_mapping="exp_mean", alpha_local=0., alpha_chunk=0.,
        temperature_local=1e-300, temperature_chunk=1e-300,
    )
    assert torch.equal(w, torch.ones_like(w))


def test_exp_is_detached_without_input_mutation_or_rng_consumption():
    v, success = _example(); v.requires_grad_(True)
    original, flags, rng = v.detach().clone(), success.clone(), torch.get_rng_state().clone()
    weights, _ = build_two_level_success_weights(v, success, factor_mapping="exp_mean")
    assert weights.dtype == torch.float32 and weights.device == v.device
    assert not weights.requires_grad and weights.grad_fn is None
    assert torch.equal(v, original) and torch.equal(success, flags)
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("name", ["temperature_local", "temperature_chunk"])
@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf")])
def test_invalid_temperature_is_rejected_before_an_empty_domain_return(name, value):
    for mapping in ("linear_centered", "exp_mean"):
        with pytest.raises(ValueError, match=name):
            build_two_level_success_weights(
                torch.ones(2, 10), torch.zeros(2, dtype=torch.bool),
                factor_mapping=mapping, **{name:value},
            )


def test_exp_rejects_unknown_mapping_nonfinite_v_and_unrepresentable_weights():
    v, success = _example()
    with pytest.raises(ValueError, match="factor_mapping"):
        build_two_level_success_weights(v, success, factor_mapping="softmax_other")
    bad = v.clone(); bad[-1, 0] = float("nan")
    with pytest.raises(ValueError, match="finite and nonnegative"):
        build_two_level_success_weights(bad, success, factor_mapping="exp_mean")
    with pytest.raises(ValueError, match="positive finite float32"):
        build_two_level_success_weights(
            v, success, factor_mapping="exp_mean",
            temperature_local=1e-300, temperature_chunk=1e-300,
        )


class _Base:
    def __init__(self, cfg):
        self.cfg, self._rank, self._world_size, self.update_step = cfg, 0, 1, 0


def _make_worker(**overrides):
    # Execute real method bodies while avoiding device/Ray/model construction.
    path = Path(__file__).resolve().parents[2] / "rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py"
    tree = ast.parse(path.read_text())
    selected = {
        "RLTACLossMixin": {
            "_rlt_dvac_selected_variances", "_prepare_global_batch",
            "_rlt_dvac_success_scale_schedule", "_effective_rlt_dvac_success_scale",
        },
        "RLTACFSDPPolicy": {"__init__", "_rlt_contract"},
    }
    classes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in selected:
            node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in selected[node.name]]
            if node.name == "RLTACFSDPPolicy":
                node.bases = [ast.Name("RLTACLossMixin", ast.Load()), ast.Name("_Base", ast.Load())]
            classes.append(node)
    from rlinf.algorithms.rlt.dvac_controls import rlt_controls_contract
    namespace = dict(rlt_controls_contract=rlt_controls_contract, torch=torch, math=math, json=json, hashlib=hashlib,
                     OmegaConf=OmegaConf, _Base=_Base,
                     build_two_level_success_weights=build_two_level_success_weights)
    exec(compile(ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[])), str(path), "exec"), namespace)
    dvac = dict(mode="apply", mapping="two_level_batch", outer_scope="successful_batch",
                alpha_local=1., alpha_chunk=1., success_scale=1.)
    dvac.update(overrides)
    cfg = OmegaConf.create(dict(
        actor=dict(global_batch_size=512, model=dict(num_action_chunks=10, action_dim=14)),
        env=dict(train=dict(auto_reset=False)),
        algorithm=dict(rlt_dvac=dvac, rlt_resume=dict(enable=True, contract=dict(baseline="clean4env"))),
    ))
    return namespace["RLTACFSDPPolicy"](cfg)


def test_worker_uses_full_b512_success_domain_and_both_temperature_parameters():
    worker = _make_worker(factor_mapping="exp_mean", temperature_local=0.5, temperature_chunk=1.)
    profile = torch.tensor([0., 1.] * 5, dtype=torch.float64)
    selected = (torch.arange(512, dtype=torch.float64)[:, None] / 128 + profile).exp()
    teacher_v = torch.full((512, 3, 50), 1e20, dtype=torch.float64)
    teacher_v[:, 1, :10] = selected
    success = torch.arange(512) % 2 == 0
    batch = dict(curr_obs=dict(teacher_dvac_v=teacher_v, episode_success=success[:, None]))
    original_cfg = copy.deepcopy(worker.rlt_dvac_cfg)
    prepared, metrics = worker._prepare_global_batch(batch, train_actor=True)
    expected, _ = build_two_level_success_weights(
        selected, success, factor_mapping="exp_mean", temperature_local=0.5, temperature_chunk=1.,
    )
    assert torch.equal(prepared["rlt_dvac_new_weights"], expected)
    assert "rlt_dvac_new_weights" not in batch and worker.rlt_dvac_cfg == original_cfg
    assert expected[success].mean().item() == pytest.approx(1.)
    assert torch.equal(expected[~success], torch.ones_like(expected[~success]))
    assert metrics["rlt_dvac_new/temperature_local"] == 0.5
    assert metrics["rlt_dvac_new/temperature_chunk"] == 1.
    unchanged, stats = worker._prepare_global_batch(batch, train_actor=False)
    assert unchanged is batch and stats == {}
    partial = dict(curr_obs={k:v[:256] for k,v in batch["curr_obs"].items()})
    with pytest.raises(ValueError, match="complete actor batch"):
        worker._prepare_global_batch(partial, train_actor=True)


@pytest.mark.parametrize("overrides", [
    {"factor_mapping":"unknown"}, {"temperature_local":0.},
    {"temperature_chunk":float("nan")}, {"temperature_local":float("inf")},
])
def test_worker_rejects_invalid_mapping_and_temperatures_at_init(overrides):
    with pytest.raises(ValueError, match="factor_mapping|temperature"):
        _make_worker(**overrides)


def test_legacy_contract_is_unchanged_and_new_temperatures_change_contract():
    legacy = _make_worker()
    expected = dict(baseline="clean4env", rlt_dvac=dict(legacy.rlt_dvac_cfg))
    serialized = json.dumps(expected, sort_keys=True, separators=(",", ":"))
    assert legacy._rlt_contract() == (serialized, hashlib.sha256(serialized.encode()).hexdigest())
    assert "factor_mapping" not in legacy.rlt_dvac_cfg
    assert "temperature_local" not in legacy.rlt_dvac_cfg
    assert legacy.rlt_dvac_factor_mapping == "linear_centered"
    assert legacy.rlt_dvac_temperature_local == legacy.rlt_dvac_temperature_chunk == 1.
    baseline = _make_worker(factor_mapping="exp_mean", temperature_local=0.5, temperature_chunk=0.5)
    for updates in (dict(temperature_local=1.), dict(temperature_chunk=1.), dict(factor_mapping="linear_centered")):
        cfg = dict(factor_mapping="exp_mean", temperature_local=0.5, temperature_chunk=0.5)
        cfg.update(updates)
        assert _make_worker(**cfg)._rlt_contract()[1] != baseline._rlt_contract()[1]
