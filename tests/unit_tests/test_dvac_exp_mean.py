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

"""Distribution/contract checks for the optional positive DVAC mapping."""

import pytest
import torch

from rlinf.algorithms.dvac_two_level import (
    canonicalize_dvac_two_level_contract,
    compute_dvac_two_level_weights,
    dvac_mapping_contract,
)


def inputs(dtype=torch.float64):
    log_v = torch.tensor(
        [
            [[0.0, 0.5, 2.0], [1.0, 2.0, 4.0], [0.5, 2.0, 3.0]],
            [[2.0, 3.0, 4.0], [0.0, 1.0, 4.0], [1.0, 1.0, 1.0]],
        ],
        dtype=dtype,
    )
    return (
        log_v.exp(),
        torch.ones(2, 3, 1, dtype=torch.bool),
        torch.tensor([3, 3, 7]),
        torch.tensor([[[1.0], [-1.0], [0.0]], [[-1.0], [2.0], [1.0]]], dtype=dtype),
    )


def compute(*args, **kwargs):
    options = dict(mapping="exp_mean", temperature_local=0.5, temperature_chunk=0.5)
    options.update(kwargs)
    return compute_dvac_two_level_weights(*args, **options)


def test_exp_matches_softmax_locally_and_loss_mass_normalized_outer():
    args = inputs()
    mass = torch.tensor([[1.0, 3.0, 2.0], [0.5, 2.0, 4.0]], dtype=torch.float64)
    weights, data = compute(*args, chunk_contributions=mass)
    x = torch.log(args[0] + 1e-12)
    z = (x - x.amin(-1, keepdim=True)) / (
        x.amax(-1, keepdim=True) - x.amin(-1, keepdim=True) + 1e-6
    )
    local_expected = 3 * torch.softmax(z / 0.5, dim=-1)
    torch.testing.assert_close(data["local_factors"], local_expected)
    for group in args[2].unique():
        selected = (args[2] == group)[None].expand(2, -1)
        score = x.mean(-1)[selected]
        unit = (score - score.min()) / (score.max() - score.min() + 1e-6)
        probability = torch.softmax(unit / 0.5, dim=0)
        g = probability * mass[selected].sum() / (probability * mass[selected]).sum()
        torch.testing.assert_close(data["chunk_factors"].squeeze(-1)[selected], g)
        torch.testing.assert_close(
            (weights.mean(-1)[selected] * mass[selected]).sum() / mass[selected].sum(),
            torch.tensor(1.0, dtype=weights.dtype),
        )
    torch.testing.assert_close(weights.mean(-1, keepdim=True), data["chunk_factors"])


@pytest.mark.parametrize("scope", ["both", "positive", "negative"])
def test_scope_mask_and_zero_mass_remain_outside_both_domains(scope):
    variance, mask, ids, adv = inputs()
    mass = torch.tensor([[1.0, 0.0, 2.0], [3.0, 2.0, 1.0]], dtype=torch.float64)
    mask[1, 2] = False
    variance[1, 2] = float("nan")
    weights, data = compute(
        variance, mask, ids, adv, chunk_contributions=mass, scope=scope
    )
    eligible = mask.squeeze(-1) & (mass > 0)
    if scope != "both":
        eligible &= adv.squeeze(-1) > 0 if scope == "positive" else adv.squeeze(-1) < 0
    assert torch.equal(data["eligible_mask"].squeeze(-1), eligible)
    assert torch.equal(weights[~eligible], torch.ones_like(weights[~eligible]))
    modified = variance.clone()
    modified[mask.squeeze(-1) & ~eligible] = 1e20
    again, _ = compute(modified, mask, ids, adv, chunk_contributions=mass, scope=scope)
    torch.testing.assert_close(weights, again)
    assert torch.isfinite(weights).all()


def test_constant_single_element_and_empty_domains_are_exact_identity():
    variance = torch.zeros(2, 3, 1)
    mask = torch.ones(2, 3, 1, dtype=torch.bool)
    ids = torch.tensor([1, 3, 9])
    adv = torch.ones_like(variance)
    for active in (mask, ~mask):
        weights, _ = compute(variance, active, ids, adv)
        assert torch.equal(weights, torch.ones_like(weights))


def test_smaller_temperature_concentrates_and_alpha_mixes_each_layer():
    args = inputs()
    warm, _ = compute(*args, alpha_chunk=0.0, temperature_local=1.0)
    cold, _ = compute(*args, alpha_chunk=0.0, temperature_local=0.25)
    assert cold.square().mean() > warm.square().mean()
    assert cold.max() > warm.max()
    mixed, _ = compute(
        *args, alpha_chunk=0.0, alpha_local=0.5, temperature_local=0.25
    )
    torch.testing.assert_close(mixed, 0.5 + 0.5 * cold)
    identity, _ = compute(*args, alpha_local=0.0, alpha_chunk=0.0)
    assert torch.equal(identity, torch.ones_like(identity))


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_mapping_is_finite_nonnegative_detached_and_preserves_compute_dtype(dtype):
    variance, mask, ids, adv = inputs(dtype)
    variance.requires_grad_()
    adv.requires_grad_()
    weights, data = compute(variance, mask, ids, adv)
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    assert weights.dtype == expected_dtype
    assert torch.isfinite(weights).all() and (weights >= 0).all()
    assert all(not value.requires_grad for value in (weights, *data.values()))


def test_tiny_temperature_underflow_is_finite_with_a_nonzero_maximum():
    weights, _ = compute(
        *inputs(torch.float32), temperature_local=1e-100, temperature_chunk=1e-100
    )
    assert torch.isfinite(weights).all() and (weights >= 0).all()
    assert weights.max() > 1


def test_shuffle_and_loss_mass_rescaling_preserve_exp_weights():
    args = inputs()
    mass = torch.tensor([[1.0, 3.0, 2.0], [0.5, 2.0, 4.0]], dtype=torch.float64)
    expected, _ = compute(*args, chunk_contributions=mass)
    order = torch.tensor([2, 0, 1])
    changed, _ = compute(
        args[0][:, order], args[1][:, order], args[2][order], args[3][:, order],
        chunk_contributions=mass[:, order] * 1e200,
    )
    torch.testing.assert_close(changed, expected[:, order])


def test_linear_default_retains_exact_results_and_ignores_inactive_temperatures():
    args = inputs()
    old, _ = compute_dvac_two_level_weights(*args)
    explicit, _ = compute_dvac_two_level_weights(
        *args, mapping="linear_centered", temperature_local=0.0,
        temperature_chunk=float("nan"),
    )
    assert torch.equal(old, explicit)
    assert dvac_mapping_contract() == {}
    legacy = {"scope": "both", "alpha_local": 0.7, "custom_future_key": 1}
    assert canonicalize_dvac_two_level_contract(legacy) == legacy
    assert canonicalize_dvac_two_level_contract(
        dict(legacy, mapping="linear_centered", temperature_local=0.2)
    ) == legacy


@pytest.mark.parametrize("field", ["temperature_local", "temperature_chunk"])
@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_effective_temperature_rejected_in_contract_and_compute(field, bad):
    with pytest.raises(ValueError, match="finite and positive"):
        dvac_mapping_contract("exp_mean", **{field: bad})
    with pytest.raises(ValueError, match="finite and positive"):
        compute(*inputs(), **{field: bad})


def test_unknown_mapping_rejected_and_exp_defaults_remain_explicit():
    with pytest.raises(ValueError, match="mapping"):
        compute(*inputs(), mapping="unknown")
    contract = canonicalize_dvac_two_level_contract({"mapping": "exp_mean"})
    assert contract == {
        "mapping": "exp_mean", "temperature_local": 1.0, "temperature_chunk": 1.0
    }
