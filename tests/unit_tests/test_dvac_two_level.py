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

"""Focused CPU checks; run only in the source-locked server environment."""

import pytest
import torch

from rlinf.algorithms.dvac_two_level import compute_dvac_two_level_weights


def _inputs(log_values, group_ids=None):
    x = torch.tensor(log_values, dtype=torch.float64)
    t, b, _ = x.shape
    return (
        x.exp(),
        torch.ones(t, b, 1, dtype=torch.bool),
        torch.zeros(b, dtype=torch.long)
        if group_ids is None
        else torch.tensor(group_ids),
        torch.ones(t, b, 1, dtype=torch.float64),
    )


def test_two_level_analytic_values_and_variance_direction():
    inputs = _inputs([[[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]]])
    weights, data = compute_dvac_two_level_weights(*inputs)
    delta = 1.0 / (2.0 + 1e-6)
    local = torch.tensor([1 - delta, 1.0, 1 + delta], dtype=torch.float64)
    outer = torch.tensor([1 - delta, 1 + delta], dtype=torch.float64)
    torch.testing.assert_close(
        data["local_factors"], local.expand(1, 2, 3), atol=1e-10, rtol=0
    )
    torch.testing.assert_close(
        data["chunk_factors"].reshape(-1), outer, atol=1e-10, rtol=0
    )
    torch.testing.assert_close(
        weights, outer[None, :, None] * local, atol=1e-10, rtol=0
    )
    assert weights[0, 0, 2] > weights[0, 0, 0]
    assert data["chunk_factors"][0, 1] > data["chunk_factors"][0, 0]


def test_outer_score_keeps_chunk_level_before_inner_normalization():
    # Both chunks have identical internal shape; a level shift must survive in g.
    inputs = _inputs([[[0.0, 0.0], [4.0, 4.0]]])
    weights, data = compute_dvac_two_level_weights(*inputs)
    assert torch.equal(data["local_factors"], torch.ones_like(weights))
    assert data["chunk_scores"][0, 1] > data["chunk_scores"][0, 0] + 3.9
    assert data["chunk_factors"][0, 1] > 1.4
    assert data["chunk_factors"][0, 0] < 0.6
    # No final per-chunk normalization is allowed to erase the outer factor.
    torch.testing.assert_close(weights.mean(-1, keepdim=True), data["chunk_factors"])


def test_zero_strength_is_exact_identity_and_every_output_is_detached():
    variance, mask, ids, adv = _inputs([[[0.0, 2.0], [1.0, 4.0]]])
    variance.requires_grad_()
    adv.requires_grad_()
    original = variance.detach().clone()
    weights, data = compute_dvac_two_level_weights(
        variance, mask, ids, adv, alpha_local=0.0, alpha_chunk=0.0
    )
    assert torch.equal(weights, torch.ones_like(weights))
    assert not weights.requires_grad
    assert all(not value.requires_grad for value in data.values())
    assert variance.requires_grad and adv.requires_grad
    torch.testing.assert_close(variance.detach(), original)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_low_precision_promotes_to_float32_and_both_includes_zero_advantage(dtype):
    variance = torch.tensor([[[1.0, 1.0], [4.0, 4.0]]], dtype=dtype, requires_grad=True)
    adv = torch.tensor([[[0.0], [-1.0]]], dtype=dtype)
    weights, data = compute_dvac_two_level_weights(
        variance, torch.ones(1, 2, dtype=torch.bool), torch.tensor([7, 7]), adv
    )
    assert weights.dtype == torch.float32 and not weights.requires_grad
    assert data["eligible_mask"].all()
    assert weights[0, 0].mean() < 1.0 and weights[0, 1].mean() > 1.0
    assert torch.equal(
        adv * weights, torch.tensor([[[0.0, 0.0], [-1.0, -1.0]]]) * weights
    )


def test_constant_and_empty_groups_are_neutral_and_invalid_values_are_ignored():
    variance = torch.tensor([[[0.0, 0.0], [float("nan"), -1.0], [float("inf"), 3.0]]])
    mask = torch.tensor([[[True], [False], [False]]])
    adv = torch.tensor([[[1.0], [float("nan")], [float("inf")]]])
    mass = torch.tensor([[2.0, float("nan"), -3.0]])
    weights, data = compute_dvac_two_level_weights(
        variance, mask, torch.tensor([9, 9, 20]), adv, chunk_contributions=mass
    )
    assert torch.equal(weights, torch.ones_like(weights))
    assert torch.isfinite(data["log_variance"]).all()
    assert torch.equal(data["eligible_mask"], mask)
    assert torch.equal(data["chunk_scores"][~mask], torch.zeros(2))
    empty, diagnostic = compute_dvac_two_level_weights(
        variance,
        torch.zeros_like(mask),
        torch.tensor([9, 9, 20]),
        adv,
        chunk_contributions=mass,
    )
    assert torch.equal(empty, torch.ones_like(empty))
    assert not diagnostic["eligible_mask"].any()


@pytest.mark.parametrize("scope", ["both", "positive", "negative"])
def test_actor_contribution_weighted_center_matches_final_loss_mass(scope):
    variance, mask, ids, adv = _inputs(
        [[[0.0, 1.0], [2.0, 4.0], [3.0, 6.0]], [[1.0, 2.0], [3.0, 5.0], [4.0, 7.0]]],
        [5, 11, 5],
    )
    adv[:, :, 0] = torch.tensor([[1.0, -1.0, -2.0], [2.0, 1.0, -1.0]])
    mask[1, 1] = False
    mass = torch.tensor([[1.0, 2.0, 4.0], [3.0, 0.0, 2.0]], dtype=torch.float64)
    weights, data = compute_dvac_two_level_weights(
        variance,
        mask,
        ids,
        adv,
        scope=scope,
        alpha_local=0.4,
        alpha_chunk=0.7,
        chunk_contributions=mass,
    )
    active = data["eligible_mask"].squeeze(-1)
    local_mean = data["local_factors"].mean(-1)
    torch.testing.assert_close(local_mean[active], torch.ones_like(local_mean[active]))
    torch.testing.assert_close(weights.mean(-1, keepdim=True), data["chunk_factors"])
    assert torch.equal(weights[~active], torch.ones_like(weights[~active]))
    for group_id in ids.unique():
        selected = active & (ids == group_id)[None, :]
        if selected.any():
            effective_mean = (weights.mean(-1)[selected] * mass[selected]).sum() / mass[
                selected
            ].sum()
            torch.testing.assert_close(
                effective_mean, torch.tensor(1.0, dtype=torch.float64)
            )


@pytest.mark.parametrize("scope,selected", [("positive", [0, 3]), ("negative", [1, 4])])
def test_partial_scope_centers_only_its_eligible_sign(scope, selected):
    inputs = list(
        _inputs([[[0.0, 0.0], [3.0, 3.0], [4.0, 4.0], [2.0, 2.0], [5.0, 5.0]]])
    )
    inputs[3] = torch.tensor(
        [[[1.0], [-1.0], [0.0], [2.0], [-2.0]]], dtype=torch.float64
    )
    weights, data = compute_dvac_two_level_weights(*inputs, scope=scope)
    expected_mask = torch.zeros(1, 5, 1, dtype=torch.bool)
    expected_mask[:, selected] = True
    assert torch.equal(data["eligible_mask"], expected_mask)
    assert weights[0, selected[0], 0] < 0.6 and weights[0, selected[1], 0] > 1.4
    assert torch.equal(
        weights[~expected_mask.expand_as(weights)], torch.ones(6, dtype=torch.float64)
    )
    # Changing opposite/zero-sign signals cannot change selected weights.
    altered = inputs[0].clone()
    altered[~expected_mask.expand_as(altered)] = 1e20
    again, _ = compute_dvac_two_level_weights(altered, *inputs[1:], scope=scope)
    torch.testing.assert_close(again, weights)


def test_zero_contribution_and_no_selected_sign_are_neutral():
    variance, mask, ids, adv = _inputs([[[0.0, 1.0], [20.0, 20.0], [2.0, 3.0]]])
    mass = torch.tensor([[1.0, 0.0, 2.0]], dtype=torch.float64)
    weights, data = compute_dvac_two_level_weights(
        variance, mask, ids, adv, chunk_contributions=mass
    )
    assert not data["eligible_mask"][0, 1, 0]
    assert torch.equal(weights[0, 1], torch.ones(2, dtype=torch.float64))
    neutral, _ = compute_dvac_two_level_weights(
        variance, mask, ids, adv, scope="negative"
    )
    assert torch.equal(neutral, torch.ones_like(neutral))


def test_two_shards_recombined_with_explicit_ids_and_shuffling_are_invariant():
    variance, mask, ids, adv = _inputs(
        [
            [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [5.0, 6.0]],
            [[1.0, 3.0], [0.0, 2.0], [6.0, 8.0], [4.0, 7.0]],
        ],
        [101, 101, 707, 707],
    )
    mask[1, 0] = False
    mass = torch.tensor(
        [[1.0, 2.0, 3.0, 1.0], [2.0, 1.0, 0.5, 4.0]], dtype=torch.float64
    )
    expected, diag = compute_dvac_two_level_weights(
        variance, mask, ids, adv, chunk_contributions=mass
    )
    # Each synthetic rank owns a complete scene group. Reverse gather order,
    # then interleave groups; IDs, not adjacency or local row index, define G.
    gather = torch.tensor([2, 3, 0, 1])
    shuffle = torch.tensor([0, 2, 1, 3])
    permutation = gather[shuffle]
    time_order = torch.tensor([1, 0])
    result, moved = compute_dvac_two_level_weights(
        variance[time_order][:, permutation],
        mask[time_order][:, permutation],
        ids[permutation],
        adv[time_order][:, permutation],
        chunk_contributions=mass[time_order][:, permutation],
    )
    torch.testing.assert_close(result, expected[time_order][:, permutation])
    torch.testing.assert_close(
        moved["chunk_factors"], diag["chunk_factors"][time_order][:, permutation]
    )
    changed = variance.clone()
    changed[:, ids == 707] *= 100.0
    separated, _ = compute_dvac_two_level_weights(
        changed, mask, ids, adv, chunk_contributions=mass
    )
    torch.testing.assert_close(separated[:, ids == 101], expected[:, ids == 101])


def test_near_constant_range_is_damped_by_minmax_epsilon():
    inputs = _inputs([[[0.0, 1e-9]]])
    weights, _ = compute_dvac_two_level_weights(*inputs, alpha_chunk=0.0)
    assert 0.0 < (weights.max() - weights.min()).item() < 0.0011
    torch.testing.assert_close(weights.mean(), torch.tensor(1.0, dtype=torch.float64))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.01])
def test_nonfinite_or_negative_actor_valid_variance_raises_even_outside_scope(bad):
    variance, mask, ids, adv = _inputs([[[0.0, 1.0]]])
    variance[0, 0, 0] = bad
    with pytest.raises(ValueError, match="actor-valid DVAC variance"):
        compute_dvac_two_level_weights(variance, mask, ids, adv, scope="negative")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha_local": -0.1},
        {"alpha_chunk": 1.01},
        {"alpha_local": float("nan")},
        {"log_eps": 0.0},
        {"minmax_eps": float("inf")},
        {"scope": "unknown"},
    ],
)
def test_invalid_numeric_contract_raises(kwargs):
    with pytest.raises(ValueError):
        compute_dvac_two_level_weights(*_inputs([[[0.0, 1.0]]]), **kwargs)


def test_rejects_action_mask_fractional_mask_and_invalid_contributions():
    variance, mask, ids, adv = _inputs([[[0.0, 1.0], [1.0, 2.0]]])
    with pytest.raises(ValueError, match="chunk mask"):
        compute_dvac_two_level_weights(variance, mask.expand_as(variance), ids, adv)
    with pytest.raises(ValueError, match="binary"):
        compute_dvac_two_level_weights(variance, mask.float() * 0.5, ids, adv)
    with pytest.raises(TypeError, match="integer IDs"):
        compute_dvac_two_level_weights(variance, mask, ids.float(), adv)
    with pytest.raises(ValueError, match="chunk_contributions"):
        compute_dvac_two_level_weights(
            variance, mask, ids, adv, chunk_contributions=torch.tensor([[1.0, -1.0]])
        )
