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

"""Focused numerical contracts; execute in the source-locked server environment."""

import math

import pytest
import torch

from rlinf.algorithms.online_bc_dvac_two_level import compute_two_level_bc_weights


def _variance(log_values):
    return torch.tensor(log_values, dtype=torch.float64).exp()


def _fm_loss(error, mask, weights=None):
    if weights is not None:
        error = error * weights[..., None]
    numerator = (error * mask).sum(dim=(1, 2))
    denominator = mask.sum(dim=(1, 2)).clamp_min(1)
    return (numerator / denominator).mean()


def test_analytic_two_levels_preserve_raw_chunk_score():
    variance = _variance([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]])
    mask = torch.ones(2, 3, 2, dtype=torch.bool)
    weights, stats = compute_two_level_bc_weights(variance, mask)
    delta = 1.0 / (2.0 + 1e-6)
    local = torch.tensor([1 - delta, 1.0, 1 + delta], dtype=torch.float64)
    outer = torch.tensor([1 - delta, 1 + delta], dtype=torch.float64)
    torch.testing.assert_close(weights, outer[:, None] * local, atol=1e-10, rtol=0)
    assert stats["weight_mean"] == pytest.approx(1.0)
    assert stats["chunk_std"] == pytest.approx(delta)
    assert weights[1].mean() > weights[0].mean()
    assert weights[0, 2] > weights[0, 0]


def test_partial_dimension_mask_uses_loss_mass_not_unweighted_action_mean():
    variance = _variance([[0.0, 2.0, 0.0], [2.0, 4.0, 6.0], [0.0, 0.0, 0.0]])
    variance[0, 2] = float("nan")
    variance[2] = torch.tensor([float("inf"), -1.0, float("nan")])
    mask = torch.tensor(
        [
            [[1, 1, 1], [1, 0, 0], [0, 0, 0]],
            [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ],
        dtype=torch.bool,
    )
    local, local_stats = compute_two_level_bc_weights(variance, mask, alpha_chunk=0)
    q = mask.sum(-1)
    delta = 2 / (2 + 1e-6)
    expected_first = torch.tensor([1 - delta / 4, 1 + 3 * delta / 4, 1.0])
    torch.testing.assert_close(local[0], expected_first.to(torch.float64))
    torch.testing.assert_close(
        (local * q).sum(-1)[:2] / q.sum(-1)[:2], torch.ones(2).double()
    )
    assert local_stats["weight_mean"] == pytest.approx(1.0)
    weights, stats = compute_two_level_bc_weights(variance, mask)
    assert stats["valid_query_count"] == 2
    assert stats["valid_action_count"] == 5
    assert stats["valid_element_count"] == 10
    assert stats["weight_mean"] == pytest.approx(1.0)
    assert torch.equal(weights[q == 0], torch.ones_like(weights[q == 0]))
    # Raw scores are 0.5 and 14/3; normalized local means are both one.
    span = 14 / 3 - 0.5
    expected_outer = torch.tensor(
        [1 - span / (2 * (span + 1e-6)), 1 + span / (2 * (span + 1e-6))],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        (weights * q).sum(-1)[:2] / q.sum(-1)[:2], expected_outer
    )
    coefficients = weights[..., None] * mask / q.sum(-1).clamp_min(1)[:, None, None]
    sumsq = coefficients.square().sum().item()
    assert stats["coefficient_sumsq"] == pytest.approx(sumsq)
    assert stats["coefficient_ess"] == pytest.approx(
        coefficients.sum().square().item() / sumsq
    )


def test_zero_strength_exactly_preserves_loss_gradient_and_detaches_signal():
    variance = _variance([[0.0, 2.0], [1.0, 4.0]]).requires_grad_()
    mask = torch.tensor([[[1, 1], [0, 0]], [[1, 0], [1, 1]]], dtype=torch.bool)
    before = variance.detach().clone()
    weights, stats = compute_two_level_bc_weights(variance, mask, 0.0, 0.0)
    assert torch.equal(weights, torch.ones_like(weights))
    assert not weights.requires_grad
    assert all(isinstance(value, float) for value in stats.values())
    prediction = torch.arange(8, dtype=torch.float64).reshape(2, 2, 2).requires_grad_()
    plain = _fm_loss(prediction.square(), mask)
    weighted = _fm_loss(prediction.square(), mask, weights)
    torch.testing.assert_close(plain, weighted, atol=0, rtol=0)
    grad_plain = torch.autograd.grad(plain, prediction, retain_graph=True)[0]
    grad_weighted = torch.autograd.grad(weighted, prediction)[0]
    torch.testing.assert_close(grad_plain, grad_weighted, atol=0, rtol=0)
    assert variance.grad is None
    torch.testing.assert_close(variance.detach(), before, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_low_precision_is_promoted_and_constant_domains_are_neutral(dtype):
    variance = torch.zeros(2, 3, dtype=dtype)
    weights, stats = compute_two_level_bc_weights(variance, torch.ones(2, 3).bool())
    assert weights.dtype == torch.float32
    assert torch.equal(weights, torch.ones_like(weights))
    assert stats["coefficient_ess"] == pytest.approx(6.0)
    assert stats["coefficient_ess_fraction"] == pytest.approx(1.0)
    assert stats["weight_sumsq"] == 6.0


@pytest.mark.parametrize("batch", [0, 2])
def test_empty_or_all_invalid_batch_is_finite_neutral(batch):
    variance = torch.full((batch, 3), float("nan"))
    weights, stats = compute_two_level_bc_weights(
        variance, torch.zeros(batch, 3, 2).bool()
    )
    assert torch.equal(weights, torch.ones_like(weights))
    assert stats["valid_query_count"] == 0
    assert stats["coefficient_ess"] == 0
    assert all(math.isfinite(value) for value in stats.values())


def test_duplicate_queries_count_with_multiplicity_and_permutation_is_equivariant():
    variance = _variance([[0.0, 1.0], [0.0, 1.0], [2.0, 3.0], [5.0, 6.0]])
    mask = torch.ones(4, 2, 3).bool()
    weights, stats = compute_two_level_bc_weights(variance, mask)
    torch.testing.assert_close(weights[0], weights[1], atol=0, rtol=0)
    order = torch.tensor([3, 1, 2, 0])
    moved, moved_stats = compute_two_level_bc_weights(variance[order], mask[order])
    torch.testing.assert_close(moved, weights[order])
    assert moved_stats == pytest.approx(stats)
    unique, _ = compute_two_level_bc_weights(variance[[0, 2, 3]], mask[[0, 2, 3]])
    assert not torch.allclose(weights[0], unique[0])


def test_same_query_outer_weight_changes_with_batch_companions():
    mask = torch.ones(2, 2).bool()
    first, _ = compute_two_level_bc_weights(_variance([[1.0, 2.0], [4.0, 5.0]]), mask)
    second, _ = compute_two_level_bc_weights(
        _variance([[1.0, 2.0], [-2.0, -1.0]]), mask
    )
    assert first[0].mean() < 1.0 < second[0].mean()
    # The internal allocation is the same; only its outer scale changes.
    torch.testing.assert_close(first[0] / first[0].mean(), second[0] / second[0].mean())


def test_full_1024_batch_weights_give_micro_partition_invariant_loss_and_gradient():
    log_v = torch.arange(1024 * 3, dtype=torch.float64).reshape(1024, 3) / 300
    variance = log_v.exp()
    mask = torch.ones(1024, 3, 2).bool()
    mask[::3, 1, 1] = False
    weights, stats = compute_two_level_bc_weights(variance, mask)
    inputs = (torch.arange(1024 * 6, dtype=torch.float64).reshape(1024, 3, 2) % 17) / 17
    scale = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    full_loss = _fm_loss((scale * inputs - 0.2).square(), mask, weights)
    full_gradient = torch.autograd.grad(full_loss, scale)[0]
    for micro_size in (32, 64, 127):
        micro_loss = torch.zeros((), dtype=torch.float64)
        for start in range(0, 1024, micro_size):
            end = min(start + micro_size, 1024)
            error = (scale * inputs[start:end] - 0.2).square()
            micro_loss = micro_loss + _fm_loss(
                error, mask[start:end], weights[start:end]
            ) * ((end - start) / 1024)
        micro_gradient = torch.autograd.grad(micro_loss, scale)[0]
        torch.testing.assert_close(micro_loss, full_loss)
        torch.testing.assert_close(micro_gradient, full_gradient)
    assert stats["weight_mean"] == pytest.approx(1.0)
    # A per-micro helper call is a different algorithm and must not look equal.
    wrong_weights, _ = compute_two_level_bc_weights(variance[:32], mask[:32])
    assert not torch.allclose(wrong_weights, weights[:32])


def test_near_constant_ranges_are_damped_by_epsilon():
    weights, _ = compute_two_level_bc_weights(
        _variance([[0.0, 1e-9]]), torch.ones(1, 2).bool()
    )
    assert 0 < (weights.max() - weights.min()).item() < 0.0011
    assert weights.mean().item() == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1])
def test_supervised_invalid_variance_raises_even_at_zero_strength(bad):
    variance = torch.tensor([[1.0, bad]])
    with pytest.raises(ValueError, match="supervised DVAC variance"):
        compute_two_level_bc_weights(variance, torch.ones(1, 2).bool(), 0, 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha_local": -0.1},
        {"alpha_chunk": 1.01},
        {"alpha_local": float("nan")},
        {"variance_eps": 0.0},
        {"variance_eps": 1e-100},
        {"range_eps": float("inf")},
    ],
)
def test_invalid_numeric_contract_raises(kwargs):
    with pytest.raises(ValueError):
        compute_two_level_bc_weights(
            torch.ones(1, 2), torch.ones(1, 2).bool(), **kwargs
        )


def test_bad_shape_mask_or_variance_type_raises():
    with pytest.raises(ValueError, match="align"):
        compute_two_level_bc_weights(torch.ones(1, 2), torch.ones(2, 1).bool())
    with pytest.raises(ValueError, match="binary"):
        compute_two_level_bc_weights(torch.ones(1, 2), torch.full((1, 2), 0.5))
    with pytest.raises(ValueError, match="binary"):
        compute_two_level_bc_weights(torch.ones(1, 2), torch.full((1, 2), float("nan")))
    with pytest.raises(TypeError, match="floating-point"):
        compute_two_level_bc_weights(torch.ones(1, 2).long(), torch.ones(1, 2).bool())
