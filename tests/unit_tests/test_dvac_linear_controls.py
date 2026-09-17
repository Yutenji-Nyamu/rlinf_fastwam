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

"""CPU checks for opt-in two-level DVAC schedules and chunk fallback."""

import copy

import pytest
import torch

from rlinf.algorithms.dvac_linear_controls import (
    apply_chunk_dropout,
    effective_linear_alphas,
    linear_controls_contract,
)


def config(**overrides):
    result = {
        "mapping": "linear_centered",
        "normalization": "two_level_group",
        "alpha_local": 1.0,
        "alpha_chunk": 1.0,
    }
    result.update(overrides)
    return result


def dropout_controls(probability=0.1, seed=42, mapping="linear_centered"):
    return linear_controls_contract(
        config(
            mapping=mapping,
            chunk_dropout={"enabled": True, "probability": probability, "seed": seed},
        )
    )


def test_disabled_controls_preserve_legacy_contract_and_weights():
    cfg = config(
        mapping="exp_mean",
        normalization="recent5",
        chunk_dropout={"enabled": False, "probability": float("nan")},
        alpha_schedule={"enabled": False},
    )
    assert linear_controls_contract(cfg) == {}
    assert linear_controls_contract({}) == {}
    weights = torch.tensor([[[0.2, 2.0], [1.5, 0.5]]])
    out, dropped = apply_chunk_dropout(
        weights, torch.ones(1, 2, 1, dtype=torch.bool), {}, runner_step=0
    )
    assert torch.equal(out, weights)
    assert out.data_ptr() == weights.data_ptr()
    assert not dropped.any()
    assert effective_linear_alphas(0.8, 0.5, {}, runner_step=99) == (0.8, 0.5)


@pytest.mark.parametrize("mapping", ["linear_centered", "exp_mean"])
def test_contract_is_normalized_copy_of_only_enabled_controls(mapping):
    cfg = config(
        mapping=mapping,
        chunk_dropout={"enabled": True},
        alpha_schedule={"enabled": True, "chunk": {"enabled": False}},
    )
    before = copy.deepcopy(cfg)
    contract = linear_controls_contract(cfg)
    assert cfg == before
    assert contract == {
        "chunk_dropout": {"enabled": True, "probability": 0.1, "seed": 42},
        "alpha_schedule": {
            "enabled": True,
            "local": {
                "enabled": True,
                "start_step": 1,
                "end_step": 10,
                "end_alpha": 0.2,
            },
        },
    }


@pytest.mark.parametrize(
    "version, expected", [(0, 1.0), (3, 11 / 15), (9, 0.2), (30, 0.2)]
)
@pytest.mark.parametrize("mapping", ["linear_centered", "exp_mean"])
def test_schedule_one_based_round_endpoints(version, expected, mapping):
    controls = linear_controls_contract(
        config(mapping=mapping, alpha_schedule={"enabled": True})
    )
    local, chunk = effective_linear_alphas(1.0, 1.0, controls, runner_step=version)
    assert local == pytest.approx(expected)
    assert chunk == pytest.approx(expected)


@pytest.mark.parametrize("mapping", ["linear_centered", "exp_mean"])
def test_schedule_layers_have_independent_start_target_and_end(mapping):
    controls = linear_controls_contract(
        config(
            mapping=mapping,
            alpha_schedule={
                "enabled": True,
                "local": {"start_step": 3, "end_step": 5, "end_alpha": 0.0},
                "chunk": {"start_step": 1, "end_step": 9, "end_alpha": 0.4},
            },
        )
    )
    assert effective_linear_alphas(1.0, 0.8, controls, runner_step=0) == (1.0, 0.8)
    assert effective_linear_alphas(1.0, 0.8, controls, runner_step=3) == pytest.approx(
        (0.5, 0.65)
    )
    assert effective_linear_alphas(1.0, 0.8, controls, runner_step=4) == pytest.approx(
        (0.0, 0.6)
    )
    assert effective_linear_alphas(1.0, 0.8, controls, runner_step=8) == (0.0, 0.4)


@pytest.mark.parametrize("mapping", ["linear_centered", "exp_mean"])
def test_schedule_can_disable_either_layer_or_both(mapping):
    for off in ("local", "chunk"):
        controls = linear_controls_contract(
            config(
                mapping=mapping,
                alpha_schedule={"enabled": True, off: {"enabled": False}},
            )
        )
        result = effective_linear_alphas(1.0, 1.0, controls, runner_step=9)
        assert result == ((1.0, 0.2) if off == "local" else (0.2, 1.0))
    assert (
        linear_controls_contract(
            config(
                mapping=mapping,
                alpha_schedule={
                    "enabled": True,
                    "local": {"enabled": False},
                    "chunk": {"enabled": False},
                },
            )
        )
        == {}
    )


@pytest.mark.parametrize("probability", [0.0, 1.0])
@pytest.mark.parametrize("mapping", ["linear_centered", "exp_mean"])
def test_dropout_probability_boundaries_respect_eligibility(probability, mapping):
    weights = torch.tensor([[[0.2, 2.0], [1.5, 0.5]]], requires_grad=True)
    eligible = torch.tensor([[[True], [False]]])
    result, dropped = apply_chunk_dropout(
        weights,
        eligible,
        dropout_controls(probability, mapping=mapping),
        runner_step=10,
    )
    assert torch.equal(dropped, eligible if probability else torch.zeros_like(eligible))
    assert torch.equal(result[:, 1], weights.detach()[:, 1])
    assert torch.equal(
        result[:, 0], torch.ones_like(result[:, 0]) if probability else weights[:, 0]
    )
    assert not result.requires_grad
    assert torch.equal(weights.detach(), torch.tensor([[[0.2, 2.0], [1.5, 0.5]]]))


@pytest.mark.parametrize("mapping", ["linear_centered", "exp_mean"])
@pytest.mark.parametrize("probability", [0.1, 0.2])
def test_dropout_is_per_chunk_with_no_rescaling_or_global_rng_effect(
    mapping, probability
):
    weights = torch.tensor([0.2, 0.8, 1.4, 2.0]).expand(50, 200, 4).clone()
    eligible = torch.ones(50, 200, 1, dtype=torch.bool)
    eligible[:, 100:] = False
    state = torch.random.get_rng_state().clone()
    result, dropped = apply_chunk_dropout(
        weights,
        eligible,
        dropout_controls(probability, mapping=mapping),
        runner_step=15,
    )
    assert torch.equal(torch.random.get_rng_state(), state)
    assert abs(dropped.sum().item() / eligible.sum().item() - probability) < 0.02
    assert not dropped[~eligible].any()
    expanded = dropped.expand_as(weights)
    assert torch.equal(result[expanded], torch.ones_like(result[expanded]))
    assert torch.equal(result[~expanded], weights[~expanded])
    assert torch.equal(weights[0, 0], torch.tensor([0.2, 0.8, 1.4, 2.0]))


@pytest.mark.parametrize("mapping", ["linear_centered", "exp_mean"])
def test_dropout_round_mask_repeats_across_updates_and_changes_next_round(mapping):
    weights = torch.full((20, 100, 3), 1.4)
    eligible = torch.ones(20, 100, 1, dtype=torch.bool)
    controls = dropout_controls(0.5, mapping=mapping)
    first = apply_chunk_dropout(weights, eligible, controls, runner_step=8)
    repeated = apply_chunk_dropout(weights.clone(), eligible, controls, runner_step=8)
    next_round = apply_chunk_dropout(weights, eligible, controls, runner_step=9)
    other_seed = apply_chunk_dropout(
        weights,
        eligible,
        dropout_controls(0.5, seed=43, mapping=mapping),
        runner_step=8,
    )
    assert torch.equal(first[0], repeated[0])
    assert torch.equal(first[1], repeated[1])
    assert not torch.equal(first[1], next_round[1])
    assert not torch.equal(first[1], other_seed[1])


@pytest.mark.parametrize(
    "overrides",
    [
        {"mapping": "unknown", "chunk_dropout": {"enabled": True}},
        {"normalization": "recent5", "alpha_schedule": {"enabled": True}},
        {"chunk_dropout": {"enabled": "true"}},
        {"chunk_dropout": {"enabled": True, "probability": float("nan")}},
        {"chunk_dropout": {"enabled": True, "probability": float("inf")}},
        {"chunk_dropout": {"enabled": True, "probability": -0.1}},
        {"chunk_dropout": {"enabled": True, "probability": 1.1}},
        {"chunk_dropout": {"enabled": True, "probability": True}},
        {"chunk_dropout": {"enabled": True, "seed": -1}},
        {"chunk_dropout": {"enabled": True, "seed": 2**63}},
        {"chunk_dropout": {"enabled": True, "seed": 1.0}},
        {"chunk_dropout": {"enabled": True, "seed": True}},
        {"alpha_schedule": {"enabled": "false"}},
        {"alpha_schedule": {"enabled": True, "local": {"enabled": 1}}},
        {"alpha_schedule": {"enabled": True, "local": {"start_step": 0}}},
        {"alpha_schedule": {"enabled": True, "local": {"end_step": 1}}},
        {"alpha_schedule": {"enabled": True, "local": {"start_step": 1.5}}},
        {"alpha_schedule": {"enabled": True, "local": {"end_step": True}}},
        {"alpha_schedule": {"enabled": True, "local": {"end_alpha": -0.1}}},
        {"alpha_schedule": {"enabled": True, "chunk": {"end_alpha": 1.1}}},
        {"alpha_schedule": {"enabled": True, "chunk": {"end_alpha": float("nan")}}},
        {"alpha_schedule": {"enabled": True}, "alpha_local": float("inf")},
        {"alpha_schedule": {"enabled": True}, "alpha_chunk": True},
    ],
)
def test_invalid_enabled_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        linear_controls_contract(config(**overrides))


@pytest.mark.parametrize("version", [-1, 0.5, True])
def test_runner_step_is_nonnegative_integer(version):
    controls = linear_controls_contract(config(alpha_schedule={"enabled": True}))
    with pytest.raises(ValueError, match="runner_step"):
        effective_linear_alphas(1.0, 1.0, controls, runner_step=version)
    with pytest.raises(ValueError, match="runner_step"):
        apply_chunk_dropout(
            torch.ones(1, 1, 2),
            torch.ones(1, 1, 1, dtype=torch.bool),
            dropout_controls(),
            runner_step=version,
        )


@pytest.mark.parametrize(
    "weights, eligible",
    [
        (torch.ones(2, 3), torch.ones(2, 3, 1, dtype=torch.bool)),
        (torch.ones(2, 3, 4, dtype=torch.int64), torch.ones(2, 3, 1, dtype=torch.bool)),
        (torch.ones(2, 3, 4), torch.ones(2, 3, dtype=torch.bool)),
        (torch.ones(2, 3, 4), torch.ones(2, 3, 1)),
    ],
)
def test_dropout_rejects_incompatible_shapes_or_types(weights, eligible):
    with pytest.raises(ValueError):
        apply_chunk_dropout(weights, eligible, dropout_controls(), runner_step=0)


@pytest.mark.parametrize("mapping", ["linear_centered", "exp_mean"])
@pytest.mark.parametrize(
    "version, expected", [(0, 1.0), (99, 100 / 199), (199, 0.0), (200, 0.0)]
)
def test_200_round_exit_schedule_is_exact(mapping, version, expected):
    controls = linear_controls_contract(
        config(
            mapping=mapping,
            alpha_schedule={
                "enabled": True,
                "local": {"end_step": 200, "end_alpha": 0.0},
                "chunk": {"end_step": 200, "end_alpha": 0.0},
            },
        )
    )
    alphas = effective_linear_alphas(1.0, 1.0, controls, runner_step=version)
    assert alphas == pytest.approx((expected, expected))
    if version >= 199:
        assert alphas == (0.0, 0.0)
