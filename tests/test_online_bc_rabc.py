# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Targeted numerical checks; run in the server project environment."""

import copy
import math

import numpy as np
import pytest
import torch

from rlinf.algorithms.online_bc_rabc import (
    RABCConfig,
    RABCStats,
    normalized_batch_weights,
    mix_normalized_batch_weights,
    raw_weights,
)


def _stats(values):
    result = RABCStats()
    result.update(torch.tensor(values, dtype=torch.float64))
    return result


def test_welford_merges_match_numpy_population_statistics():
    values = np.array([-4.0, -2.0, -3.0, 0.2, 0.01, 1.5, 2.0], dtype=np.float64)
    stats = RABCStats()
    for chunk in np.split(values, [2, 3, 6]):
        stats.update(chunk)
    assert stats.count == values.size
    assert stats.raw_mean == pytest.approx(values.mean(), abs=1e-15)
    assert stats.population_std == pytest.approx(values.std(ddof=0), abs=1e-15)
    assert stats.m2 == pytest.approx(((values - values.mean()) ** 2).sum(), abs=1e-14)
    # The mapping readout must not corrupt the negative running mean.
    before = stats.state_dict()
    assert stats.mapping_moments(RABCConfig(1.0))[0] == 0.0
    assert stats.state_dict() == before


def test_statistics_are_float64_and_update_is_transactional():
    stats = _stats([1.0, 2.0])
    before = stats.state_dict()
    for invalid in (torch.tensor([float("nan")]), torch.tensor([[1.0]]), [1e308, -1e308]):
        with pytest.raises(ValueError):
            stats.update(invalid)
        assert stats.state_dict() == before
    stats.update(torch.empty(0, dtype=torch.float64))
    assert stats.state_dict() == before


def test_hard_guards_exact_kappa_and_detached_weights():
    # Population mean/std = .5/.5, so the middle soft weights are known.
    stats = _stats([0.0, 1.0])
    config = RABCConfig(kappa_seconds=1.0)
    values = torch.tensor(
        [-0.4, 0.0, 0.2, 0.5, 1.0, math.nextafter(1.0, math.inf), 1.2],
        dtype=torch.float64,
        requires_grad=True,
    )
    before = stats.state_dict()
    result = raw_weights(values, stats, config)
    expected = torch.tensor(
        [0.0, 0.5 / 2.000001, 0.7 / 2.000001, 1.0 / 2.000001, 1.5 / 2.000001, 1.0, 1.0],
        dtype=torch.float64,
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=1e-15)
    assert not result.requires_grad
    assert result.dtype == torch.float64
    assert result.device == values.device
    assert stats.state_dict() == before


@pytest.mark.parametrize("constant,expected", [(0.0, 0.4), (0.005, 0.4), (2.0, 1.0), (-2.0, 0.0)])
def test_constant_signal_matches_official_std_floor(constant, expected):
    values = torch.full((5,), constant, dtype=torch.float64)
    stats = _stats(values.tolist())
    weights = raw_weights(values, stats, RABCConfig(1.0))
    torch.testing.assert_close(weights, torch.full_like(weights, expected), atol=1e-12, rtol=0)


def test_full_batch_normalization_diagnostics_and_ess():
    config = RABCConfig(1.0)
    raw = torch.tensor([0.0, 0.5, 0.5], dtype=torch.float64, requires_grad=True)
    weights, info = normalized_batch_weights(raw, config)
    expected = torch.tensor([0.0, 1.5 / 1.000001, 1.5 / 1.000001], dtype=torch.float64)
    torch.testing.assert_close(weights, expected, rtol=0, atol=1e-15)
    assert not weights.requires_grad
    assert info["num_zero_weight"] == 1
    assert info["num_full_weight"] == 0
    assert info["effective_sample_size"] == 2.0
    assert info["raw_weight_mean"] == pytest.approx(1 / 3)
    assert info["raw_weight_std"] == pytest.approx(np.std([0, 0.5, 0.5]))
    assert not info["skip_update"]
    assert weights.max() > 1  # No second clip after normalization.


def test_zero_weights_require_explicit_optimizer_skip():
    weights, info = normalized_batch_weights(torch.zeros(4), RABCConfig(1.0))
    assert torch.count_nonzero(weights) == 0
    assert info["all_zero"] and info["skip_update"]
    assert info["effective_sample_size"] == 0


def test_microbatch_accumulation_matches_full_weighted_masked_objective_gradient():
    batch, horizon, dims, micro = 12, 50, 14, 3
    config = RABCConfig(1.0)
    raw = torch.tensor(
        [0.0, 0.0, 0.1, 0.2, 0.5, 0.0, 0.6, 1.0, 1.0, 0.1, 0.2, 0.3],
        dtype=torch.float64,
    )
    normalized, _ = normalized_batch_weights(raw, config)
    target = torch.linspace(
        -1, 2, batch * horizon * dims, dtype=torch.float64
    ).reshape(batch, horizon, dims)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[::2, 30:] = False
    parameter = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    errors = (parameter - target).square()
    per_query = (errors * mask).sum((1, 2)) / mask.sum((1, 2))
    full = (raw * per_query).sum() / (raw.sum() + config.epsilon_weight)
    expected_gradient = torch.autograd.grad(full, parameter)[0]

    micro_parameter = parameter.detach().clone().requires_grad_()
    accumulated_loss = 0.0
    for start in range(0, batch, micro):
        stop = start + micro
        part_mask = mask[start:stop]
        part_errors = (micro_parameter - target[start:stop]).square()
        part_losses = (part_errors * part_mask).sum((1, 2)) / part_mask.sum((1, 2))
        contribution = (normalized[start:stop] * part_losses).mean() * (micro / batch)
        contribution.backward()
        accumulated_loss += contribution.detach().item()
    assert accumulated_loss == pytest.approx(full.item(), abs=1e-14)
    torch.testing.assert_close(micro_parameter.grad, expected_gradient, rtol=0, atol=1e-14)


def test_unit_conversion_scales_signal_epsilon_and_kappa_together():
    values = torch.tensor([-0.002, 0, 0.005, 0.01, 0.0101], dtype=torch.float64)
    reference = torch.tensor([0.0, 0.01], dtype=torch.float64)
    stats = _stats(reference.tolist())
    config = RABCConfig(0.01)
    scale = 90.0
    scaled_stats = _stats((reference * scale).tolist())
    scaled_config = RABCConfig(
        config.kappa_seconds * scale,
        config.epsilon_signal * scale,
        config.epsilon_weight,
    )
    original = raw_weights(values, stats, config)
    converted = raw_weights(values * scale, scaled_stats, scaled_config)
    torch.testing.assert_close(converted, original, rtol=0, atol=1e-15)
    normalized, _ = normalized_batch_weights(original, config)
    normalized_converted, _ = normalized_batch_weights(converted, scaled_config)
    torch.testing.assert_close(normalized_converted, normalized, rtol=0, atol=1e-15)


def test_checkpoint_restores_moments_and_future_updates_exactly():
    original = _stats([-1.0, 0.5, 2.0])
    restored = RABCStats()
    restored.load_state_dict(copy.deepcopy(original.state_dict()))
    original.update([0.1, 0.2])
    restored.update([0.1, 0.2])
    assert restored.state_dict() == original.state_dict()
    query = torch.tensor([0.0, 0.3, 1.5])
    torch.testing.assert_close(
        raw_weights(query, restored, RABCConfig(1.0)),
        raw_weights(query, original, RABCConfig(1.0)),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "key,value",
    [("format", "dvac"), ("version", 2), ("version", True), ("units", "normalized"),
     ("variance", "sample"), ("count", -1), ("count", True), ("m2", -1.0),
     ("raw_mean", float("nan")), ("m2", float("inf"))],
)
def test_checkpoint_rejects_bad_identity_or_moments_transactionally(key, value):
    stats = _stats([0.0, 1.0])
    original = stats.state_dict()
    bad = {**original, key: value}
    with pytest.raises(ValueError):
        stats.load_state_dict(bad)
    assert stats.state_dict() == original


def test_checkpoint_requires_exact_schema_and_consistent_empty_state():
    stats = RABCStats()
    original = stats.state_dict()
    invalid_states = (
        {**original, "old_history": []},
        {k: v for k, v in original.items() if k != "m2"},
        {**original, "raw_mean": 1.0},
    )
    for state in invalid_states:
        with pytest.raises(ValueError):
            stats.load_state_dict(state)
        assert stats.state_dict() == original


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kappa_seconds": 0},
        {"kappa_seconds": float("nan")},
        {"kappa_seconds": True},
        {"kappa_seconds": 1, "epsilon_signal": 0},
        {"kappa_seconds": 1, "epsilon_weight": -1},
    ],
)
def test_config_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        RABCConfig(**kwargs)


@pytest.mark.parametrize("values", [[], [[0.2]], [float("nan")], [float("inf")]])
def test_weight_input_is_nonempty_finite_vector(values):
    with pytest.raises(ValueError):
        raw_weights(values, _stats([0, 1]), RABCConfig(1))


def test_raw_normalization_rejects_out_of_range_weights_and_empty_stats():
    for values in ([-0.1, 0.5], [0.1, 1.1]):
        with pytest.raises(ValueError):
            normalized_batch_weights(values, RABCConfig(1))
    with pytest.raises(ValueError):
        raw_weights([0.1], RABCStats(), RABCConfig(1))


@pytest.mark.parametrize("value", [-0.1, 1.1, True, False, float("nan"), float("inf"), "0.5", None])
def test_clean_mix_rejects_invalid_parameters(value):
    with pytest.raises(ValueError, match="clean_mix"):
        RABCConfig(2.0, clean_mix=value)


def test_mix_preserves_legacy_tensor_and_unclipped_epsilon_normalization():
    raw = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    normalized, _ = normalized_batch_weights(raw, RABCConfig(2.0))
    legacy, info0 = mix_normalized_batch_weights(normalized, RABCConfig(2.0))
    assert legacy.data_ptr() == normalized.data_ptr()
    torch.testing.assert_close(legacy, normalized, rtol=0, atol=0)
    mixed, info = mix_normalized_batch_weights(normalized, RABCConfig(2.0, clean_mix=0.5))
    torch.testing.assert_close(mixed, 0.5 + 0.5 * normalized, rtol=0, atol=0)
    assert mixed.max() > 2.5  # No post-mix clipping or normalization.
    assert mixed.mean() < 1.0  # Preserve the original epsilon denominator.
    assert info["final_weight_zero_fraction"] == 0
    assert not info["skip_update"] and not info["clean_fallback"]
    clean, _ = mix_normalized_batch_weights(normalized, RABCConfig(2.0, clean_mix=1.0))
    torch.testing.assert_close(clean, torch.ones_like(clean), rtol=0, atol=0)


@pytest.mark.parametrize("clean_mix", [0.0, 0.5, 1.0])
def test_all_zero_mix_policy_is_explicit(clean_mix):
    config = RABCConfig(2.0, clean_mix=clean_mix)
    normalized, original = normalized_batch_weights(torch.zeros(4), config)
    result, info = mix_normalized_batch_weights(normalized, config)
    assert original["skip_update"] and info["base_skip_update"]
    expected = torch.zeros(4, dtype=torch.float64) if clean_mix == 0 else torch.ones(4, dtype=torch.float64)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert info["skip_update"] == (clean_mix == 0)
    assert info["clean_fallback"] == (clean_mix > 0)
    assert info["final_effective_sample_size"] == (0 if clean_mix == 0 else 4)


@pytest.mark.parametrize("value", [torch.tensor([]), torch.tensor([float("nan")]),
                                 torch.tensor([-1.0]), torch.tensor([[1.0]]), torch.tensor([True])])
def test_mix_does_not_turn_invalid_normalized_input_into_clean(value):
    with pytest.raises(ValueError):
        mix_normalized_batch_weights(value, RABCConfig(2.0, clean_mix=0.5))
