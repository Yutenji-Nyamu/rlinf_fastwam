# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import math

import pytest
import torch

from rlinf.algorithms.rlt.dvac_two_level import build_two_level_success_weights


def _example():
    # Two successes with the same local shape, but different chunk signals.
    variances = torch.exp(torch.tensor([[0.0, 1.0], [2.0, 3.0], [20.0, -20.0]]))
    return variances, torch.tensor([True, True, False])


def test_known_two_level_mapping_and_failure_identity():
    variances, success = _example()
    weights, metrics = build_two_level_success_weights(variances, success)

    torch.testing.assert_close(
        weights, torch.tensor([[0.25, 0.75], [0.75, 2.25], [1.0, 1.0]])
    )
    assert metrics["rlt_dvac_new/success_count"] == 2.0
    assert metrics["rlt_dvac_new/success_ratio"] == pytest.approx(2 / 3)
    for domain in ("inner", "outer", "success_applied", "applied"):
        assert metrics[f"rlt_dvac_new/{domain}_mean"] == pytest.approx(1.0)
        assert 0.0 < metrics[f"rlt_dvac_new/{domain}_ess_ratio"] <= 1.0
    assert metrics["rlt_dvac_new/outer_min"] == pytest.approx(0.5)
    assert metrics["rlt_dvac_new/outer_max"] == pytest.approx(1.5)


def test_scale_changes_only_successful_rows_after_allocation():
    variances, success = _example()
    baseline, _ = build_two_level_success_weights(variances, success)
    scaled, metrics = build_two_level_success_weights(
        variances, success, success_scale=2.5
    )

    torch.testing.assert_close(scaled[success], 2.5 * baseline[success])
    assert torch.equal(scaled[~success], torch.ones_like(scaled[~success]))
    assert metrics["rlt_dvac_new/inner_mean"] == pytest.approx(1.0)
    assert metrics["rlt_dvac_new/outer_mean"] == pytest.approx(1.0)
    assert metrics["rlt_dvac_new/success_applied_mean"] == pytest.approx(2.5)
    assert metrics["rlt_dvac_new/applied_mean"] == pytest.approx(2.0)


def test_local_and_outer_can_be_disabled_independently():
    variances, success = _example()
    local_only, _ = build_two_level_success_weights(
        variances, success, alpha_chunk=0
    )
    outer_only, _ = build_two_level_success_weights(
        variances, success, alpha_local=0
    )
    identity, _ = build_two_level_success_weights(
        variances, success, alpha_local=0, alpha_chunk=0
    )

    torch.testing.assert_close(local_only[:2], torch.tensor([[0.5, 1.5]]).repeat(2, 1))
    torch.testing.assert_close(outer_only[:2], torch.tensor([[0.5, 0.5], [1.5, 1.5]]))
    assert torch.equal(identity, torch.ones_like(identity))


def test_outer_signal_uses_raw_mean_log_v_before_local_normalization():
    variances = torch.exp(torch.tensor([[0.0, 1.0], [4.0, 4.0]]))
    weights, _ = build_two_level_success_weights(
        variances, torch.tensor([True, True])
    )

    # A constant local signal is neutral internally, but remains high externally.
    torch.testing.assert_close(weights, torch.tensor([[0.25, 0.75], [1.5, 1.5]]))


def test_failure_values_do_not_set_success_comparison_domain():
    variances, success = _example()
    expected, _ = build_two_level_success_weights(variances, success)
    variances[-1] = torch.tensor([0.0, 1e30])
    actual, _ = build_two_level_success_weights(variances, success)
    assert torch.equal(expected, actual)


@pytest.mark.parametrize("shape", [(0, 10), (3, 10)])
def test_empty_success_domain_is_finite_identity(shape):
    weights, metrics = build_two_level_success_weights(
        torch.zeros(shape), torch.zeros(shape[0], dtype=torch.bool)
    )
    assert torch.equal(weights, torch.ones(shape))
    assert metrics["rlt_dvac_new/success_count"] == 0
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["rlt_dvac_new/outer_mean"] == 1


def test_one_success_has_no_outer_allocation():
    variances, _ = _example()
    success = torch.tensor([False, True, False])
    weights, metrics = build_two_level_success_weights(variances, success)
    torch.testing.assert_close(weights[1], torch.tensor([0.5, 1.5]))
    assert torch.equal(weights[~success], torch.ones(2, 2))
    assert metrics["rlt_dvac_new/outer_std"] == 0


@pytest.mark.parametrize("value", [0.0, 1.0, 1e20])
def test_constant_signal_returns_neutral_factors(value):
    weights, _ = build_two_level_success_weights(
        torch.full((3, 10), value), torch.ones(3, dtype=torch.bool)
    )
    assert torch.equal(weights, torch.ones_like(weights))


def test_near_constant_domains_use_explicit_threshold():
    variances = torch.exp(
        torch.tensor([[0.0, 1e-8], [2e-8, 3e-8]], dtype=torch.float64)
    )
    weights, _ = build_two_level_success_weights(
        variances, torch.ones(2, dtype=torch.bool), minmax_eps=1e-6
    )
    assert torch.equal(weights, torch.ones_like(weights))


def test_one_action_has_neutral_local_factor():
    weights, metrics = build_two_level_success_weights(
        torch.tensor([[1.0], [2.0]]), torch.tensor([True, True])
    )
    torch.testing.assert_close(weights, torch.tensor([[0.5], [1.5]]))
    assert metrics["rlt_dvac_new/inner_std"] == 0


def test_shuffle_equivariance_and_duplicate_multiplicity():
    variances, success = _example()
    order = torch.tensor([2, 0, 1])
    original, _ = build_two_level_success_weights(variances, success)
    shuffled, _ = build_two_level_success_weights(variances[order], success[order])
    torch.testing.assert_close(shuffled, original[order])

    duplicate_rows = torch.tensor([0, 0, 1, 2])
    duplicated, metrics = build_two_level_success_weights(
        variances[duplicate_rows], success[duplicate_rows]
    )
    # Duplicate queries count twice: outer values become 2/3, 2/3, 5/3.
    torch.testing.assert_close(
        duplicated,
        torch.tensor([[1 / 3, 1.0], [1 / 3, 1.0], [5 / 6, 2.5], [1.0, 1.0]]),
    )
    assert metrics["rlt_dvac_new/outer_mean"] == pytest.approx(1.0)


def test_does_not_modify_inputs_create_gradients_or_consume_rng():
    variances, success = _example()
    variances.requires_grad_(True)
    original = variances.detach().clone()
    flags = success.clone()
    rng_before = torch.get_rng_state().clone()
    weights, _ = build_two_level_success_weights(variances, success)

    assert not weights.requires_grad
    assert weights.grad_fn is None
    assert weights.dtype == torch.float32
    assert weights.device == variances.device
    assert torch.equal(variances, original)
    assert torch.equal(success, flags)
    assert torch.equal(rng_before, torch.get_rng_state())


@pytest.mark.parametrize("micro_sizes", [(2, 2, 2), (1, 3, 2), (6,)])
def test_precomputed_weights_preserve_loss_and_gradient_across_microbatches(
    micro_sizes,
):
    variances = torch.exp(torch.arange(60, dtype=torch.float64).reshape(6, 10) / 10)
    success = torch.tensor([True, False, True, True, False, True])
    weights, _ = build_two_level_success_weights(variances, success)
    target = torch.linspace(-1, 1, 6 * 10 * 14, dtype=torch.float64).reshape(6, 10, 14)
    parameter = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    error = (parameter - target).square().mean(dim=-1)
    full_loss = (weights * error).mean()
    full_gradient = torch.autograd.grad(full_loss, parameter)[0]

    parameter = parameter.detach().requires_grad_(True)
    loss_parts = []
    start = 0
    for size in micro_sizes:
        end = start + size
        error = (parameter - target[start:end]).square().mean(dim=-1)
        loss = (weights[start:end] * error).mean() * size / target.shape[0]
        loss.backward()
        loss_parts.append(loss.detach())
        start = end
    torch.testing.assert_close(sum(loss_parts), full_loss.detach())
    torch.testing.assert_close(parameter.grad, full_gradient)


def test_extreme_valid_log_offset_and_variances_remain_finite():
    variances = torch.tensor([[0.0, 1e-300], [1e250, 1e300]], dtype=torch.float64)
    weights, metrics = build_two_level_success_weights(
        variances, torch.tensor([True, True]), log_eps=1e-300
    )
    assert torch.isfinite(weights).all()
    assert all(math.isfinite(value) for value in metrics.values())


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
@pytest.mark.parametrize("failed_row", [False, True])
def test_invalid_variance_is_rejected_even_in_failed_queries(bad, failed_row):
    variances, success = _example()
    variances[2 if failed_row else 0, 0] = bad
    with pytest.raises(ValueError, match="finite and nonnegative"):
        build_two_level_success_weights(variances, success)


@pytest.mark.parametrize(
    "variances,success",
    [
        (torch.ones(2), torch.ones(2, dtype=torch.bool)),
        (torch.ones(2, 0), torch.ones(2, dtype=torch.bool)),
        (torch.ones(2, 2, dtype=torch.int64), torch.ones(2, dtype=torch.bool)),
        (torch.ones(2, 2), torch.ones(2, dtype=torch.float32)),
        (torch.ones(2, 2), torch.ones(2, 1, dtype=torch.bool)),
        (torch.ones(2, 2), torch.ones(3, dtype=torch.bool)),
    ],
)
def test_invalid_shapes_and_flag_dtype_are_rejected(variances, success):
    with pytest.raises(ValueError):
        build_two_level_success_weights(variances, success)


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("alpha_local", -0.1),
        ("alpha_local", 1.1),
        ("alpha_chunk", -0.1),
        ("alpha_chunk", 1.1),
        ("alpha_local", float("nan")),
        ("alpha_chunk", float("inf")),
        ("log_eps", 0.0),
        ("log_eps", float("nan")),
        ("minmax_eps", -1.0),
        ("minmax_eps", float("inf")),
        ("success_scale", 0.0),
        ("success_scale", float("nan")),
    ],
)
def test_invalid_parameters_are_rejected(parameter, value):
    variances, success = _example()
    with pytest.raises(ValueError, match=parameter):
        build_two_level_success_weights(variances, success, **{parameter: value})
