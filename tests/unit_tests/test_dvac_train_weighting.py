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

from __future__ import annotations

import pytest
import torch

from rlinf.algorithms.dvac_train_weighting import (
    DVACRecentStats,
    DVACStepStats,
    compute_endpoint_variance,
    local_log_v_sufficient_statistics,
    straight_through_scale_logprobs,
)
from rlinf.algorithms.utils import preprocess_loss_inputs
from rlinf.utils.nested_dict_process import process_nested_dict_for_train


def _recent() -> DVACRecentStats:
    return DVACRecentStats(
        window_steps=5,
        warmup_steps=1,
        log_eps=1e-12,
        std_floor=1e-6,
        z_clip=2.0,
        strength=0.5,
    )


def _recent_w0to5() -> DVACRecentStats:
    return DVACRecentStats(
        window_steps=5,
        warmup_steps=1,
        log_eps=1e-12,
        std_floor=1e-6,
        z_clip=2.0,
        strength=0.5,
        weight_min=0.0,
        weight_max=5.0,
    )


def test_endpoint_variance_uses_population_tail_variance() -> None:
    z = torch.tensor([[[[0.0]], [[1.0]], [[3.0]], [[7.0]]]])
    assert torch.allclose(compute_endpoint_variance(z, 3), torch.tensor([[56 / 9]]))


def test_straight_through_keeps_values_and_scales_gradients() -> None:
    logprobs = torch.arange(6.0).reshape(1, 3, 2).requires_grad_()
    weights = torch.tensor([[0.0, 1.0, 2.0]])
    scaled = straight_through_scale_logprobs(logprobs, weights)
    assert torch.equal(scaled, logprobs)
    scaled.sum().backward()
    assert torch.equal(
        logprobs.grad,
        weights.unsqueeze(-1).expand_as(logprobs),
    )


def test_recent_history_is_completed_step_only_and_roundtrips() -> None:
    recent = _recent()
    variance = torch.exp(torch.tensor([[[0.0, 1.0, 4.0]]]))
    weights, _, warmup, history = recent.compute_weights(variance)
    assert warmup and history["history_steps"] == 0
    assert torch.equal(weights, torch.ones_like(weights))

    recent.push(DVACStepStats(1, 2, 1.0, 1.0))
    weights, clipped_z, warmup, history = recent.compute_weights(variance)
    assert not warmup and history["history_steps"] == 1
    assert torch.allclose(clipped_z, torch.tensor([[[-1.0, 1.0, 2.0]]]))
    assert torch.allclose(weights, torch.tensor([[[0.5, 1.5, 2.0]]]))

    restored = _recent()
    restored.load_state_dict(recent.state_dict())
    assert restored.state_dict() == recent.state_dict()


def test_explicit_weight_endpoints_use_piecewise_mapping_and_roundtrip() -> None:
    recent = _recent_w0to5()
    recent.push(DVACStepStats(1, 2, 0.0, 2.0))
    variance = torch.exp(torch.tensor([[[-2.0, -1.0, 0.0, 1.0, 2.0]]]))
    weights, clipped_z, warmup, _ = recent.compute_weights(variance)
    assert not warmup
    assert torch.allclose(clipped_z, torch.tensor([[[-2.0, -1.0, 0.0, 1.0, 2.0]]]))
    assert torch.allclose(weights, torch.tensor([[[0.0, 0.5, 1.0, 3.0, 5.0]]]))

    logprobs = torch.ones(1, 5, 1, requires_grad=True)
    straight_through_scale_logprobs(logprobs, weights.squeeze(0)).sum().backward()
    assert torch.allclose(logprobs.grad.squeeze(-1), weights.squeeze(0))

    restored = _recent_w0to5()
    restored.load_state_dict(recent.state_dict())
    assert restored.state_dict() == recent.state_dict()

    incompatible = _recent()
    with pytest.raises(ValueError, match="weight_min"):
        incompatible.load_state_dict(recent.state_dict())


def test_resume_rejects_different_weighting_config() -> None:
    recent = _recent()
    state = recent.state_dict()
    state["strength"] = 0.1
    with pytest.raises(ValueError, match="strength"):
        recent.load_state_dict(state)


def test_current_nested_shuffle_keeps_weights_aligned() -> None:
    marker = torch.arange(6).reshape(2, 3, 1)
    weights = marker.expand(2, 3, 2).float()
    batch = {
        "prev_logprobs": torch.zeros(2, 3, 2, 1),
        "forward_inputs": {
            "marker": marker.clone(),
            "dvac_weights": weights,
        },
    }
    shuffled = process_nested_dict_for_train(batch, torch.tensor([5, 0, 3, 2, 1, 4]))
    assert torch.equal(
        shuffled["forward_inputs"]["dvac_weights"][:, 0].long(),
        shuffled["forward_inputs"]["marker"][:, 0],
    )


def test_action_level_dvac_weights_form_explicit_advantages() -> None:
    logprobs = torch.arange(12.0).reshape(2, 3, 2)
    old_logprobs = logprobs - 0.1
    advantages = torch.tensor([[2.0], [-1.0]], requires_grad=True)
    weights = torch.tensor(
        [[0.5, 1.0, 2.0], [2.0, 1.0, 0.5]],
        requires_grad=True,
    )

    result = preprocess_loss_inputs(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        logprob_type="action_level",
        single_action_dim=2,
        reward_type="chunk_level",
        dvac_advantage_weights=weights,
    )

    assert result["logprobs"].shape == (2, 3)
    assert torch.equal(result["advantages"], advantages * weights.detach())
    result["advantages"].sum().backward()
    assert weights.grad is None
    assert torch.equal(advantages.grad, weights.detach().sum(dim=1, keepdim=True))


def test_dvac_advantage_weights_require_action_level_logprobs() -> None:
    with pytest.raises(ValueError, match="require action_level"):
        preprocess_loss_inputs(
            logprobs=torch.zeros(1, 2, 1),
            old_logprobs=torch.zeros(1, 2, 1),
            advantages=torch.ones(1, 1),
            logprob_type="chunk_level",
            single_action_dim=1,
            reward_type="chunk_level",
            dvac_advantage_weights=torch.ones(1, 2),
        )


def test_sufficient_statistics_match_log_values() -> None:
    variance = torch.exp(torch.tensor([[[0.0, 1.0], [2.0, 3.0]]]))
    stats = local_log_v_sufficient_statistics(variance, log_eps=1e-12)
    assert torch.allclose(stats, torch.tensor([4.0, 6.0, 14.0], dtype=torch.float64))
