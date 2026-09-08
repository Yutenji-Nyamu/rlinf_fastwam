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

"""Test the native chunk loss and its gradients at the DVAC integration seam."""

import pytest
import torch

from rlinf.algorithms.losses import compute_ppo_actor_loss
from rlinf.algorithms.utils import preprocess_loss_inputs


def native_loss(
    logprobs,
    old_logprobs,
    advantages,
    weights=None,
    masked=False,
    numeric_clamp=False,
    dual_clip=False,
):
    batch_size = len(logprobs)
    mask = None
    mask_sum = None
    if masked:
        mask = torch.ones(batch_size, 1, dtype=torch.bool)
        mask[-1] = False
        # Different episode denominators exercise the unchanged masked reducer.
        mask_sum = torch.linspace(50, 200, batch_size).unsqueeze(-1)
    inputs = preprocess_loss_inputs(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        logprob_type="chunk_level",
        reward_type="chunk_level",
        single_action_dim=logprobs.shape[-1],
        loss_mask=mask,
        loss_mask_sum=mask_sum,
        dvac_chunk_advantage_weights=weights,
    )
    return compute_ppo_actor_loss(
        **inputs,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=1.5 if dual_clip else None,
        clip_log_ratio_min=-0.25 if numeric_clamp else None,
        clip_log_ratio_max=0.25 if numeric_clamp else None,
        max_episode_steps=200 if masked else None,
    )


def sample_inputs(horizon):
    # Each advantage sign sees an unclipped, PPO-clipped, and dual-clipped case.
    ratios = torch.tensor([1.05, 1.4, 0.7, 1.05, 0.7, 4.0, 1.1, 1.05])
    advantages = torch.tensor(
        [[1.0], [1.0], [1.0], [-1.0], [-1.0], [-1.0], [0.0], [1.0]]
    )
    logprobs = (
        (ratios.log()[:, None, None] / (horizon * 14))
        .expand(-1, horizon, 14)
        .clone()
        .requires_grad_()
    )
    return logprobs, advantages


@pytest.mark.parametrize("horizon", [2, 50])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("numeric_clamp", [False, True])
@pytest.mark.parametrize("dual_clip", [False, True])
def test_all_ones_match_control_loss_metrics_and_gradient(
    horizon, masked, numeric_clamp, dual_clip
):
    actual, advantages = sample_inputs(horizon)
    control = actual.detach().clone().requires_grad_()
    options = {"masked": masked, "numeric_clamp": numeric_clamp, "dual_clip": dual_clip}
    loss, metrics = native_loss(
        actual,
        torch.zeros_like(actual),
        advantages,
        torch.ones(actual.shape[:2]),
        **options,
    )
    control_loss, control_metrics = native_loss(
        control, torch.zeros_like(control), advantages, **options
    )
    torch.testing.assert_close(loss, control_loss, atol=2e-6, rtol=2e-6)
    assert metrics.keys() == control_metrics.keys()
    for name in metrics:
        torch.testing.assert_close(
            metrics[name], control_metrics[name], equal_nan=True, atol=2e-6, rtol=2e-6
        )
    loss.backward()
    control_loss.backward()
    torch.testing.assert_close(actual.grad, control.grad, atol=2e-6, rtol=2e-6)
    assert torch.count_nonzero(actual.grad[1]) == 0  # positive PPO clip
    assert torch.count_nonzero(actual.grad[4]) == 0  # negative PPO clip
    assert torch.count_nonzero(actual.grad[6]) == 0  # zero advantage
    if dual_clip or numeric_clamp:
        assert torch.count_nonzero(actual.grad[5]) == 0


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("numeric_clamp", [False, True])
@pytest.mark.parametrize("dual_clip", [False, True])
def test_both_signs_keep_local_distribution_and_chunk_strength(
    masked, numeric_clamp, dual_clip
):
    actual, advantages = sample_inputs(50)
    control = actual.detach().clone().requires_grad_()
    local_factor = torch.linspace(0.5, 1.5, 50).expand(len(actual), -1)
    chunk_factor = torch.tensor([0.6, 0.7, 0.8, 1.4, 1.3, 1.2, 0.9, 1.1])[:, None]
    weights = (local_factor * chunk_factor).clone().requires_grad_()
    options = {"masked": masked, "numeric_clamp": numeric_clamp, "dual_clip": dual_clip}
    loss, metrics = native_loss(
        actual, torch.zeros_like(actual), advantages, weights, **options
    )
    control_loss, control_metrics = native_loss(
        control, torch.zeros_like(control), advantages * chunk_factor, **options
    )
    # A chunk's forward objective keeps its mean weight g; L still changes its
    # individual action gradients, including negative advantages.
    torch.testing.assert_close(loss, control_loss, atol=2e-6, rtol=2e-6)
    for name in metrics:
        torch.testing.assert_close(
            metrics[name], control_metrics[name], equal_nan=True, atol=2e-6, rtol=2e-6
        )
    loss.backward()
    control_loss.backward()
    torch.testing.assert_close(
        actual.grad, control.grad * local_factor.unsqueeze(-1), atol=2e-6, rtol=2e-6
    )
    assert weights.grad is None
    assert actual.grad[0, 0, 0] != actual.grad[0, -1, 0]
    assert actual.grad[3, 0, 0] != actual.grad[3, -1, 0]


def test_shared_parameters_receive_weighted_action_contributions():
    torch.manual_seed(4)
    features = torch.randn(5, 3)
    model = torch.nn.Linear(3, 50 * 14, bias=False)
    logprobs = model(features).reshape(5, 50, 14) * 0.0001
    advantages = torch.tensor([[1.0], [-1.0], [1.0], [0.0], [-1.0]])
    weights = torch.rand(5, 50) + 0.5
    loss, _ = native_loss(logprobs, logprobs.detach().clone(), advantages, weights)
    actual = torch.autograd.grad(loss, model.weight, retain_graph=True)[0]
    explicit_unclipped = -(
        advantages.flatten() * (logprobs * weights[..., None]).sum((1, 2))
    ).mean()
    expected = torch.autograd.grad(explicit_unclipped, model.weight)[0]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("weights", [[0.0, 2.0], [1.3, 1.3]])
def test_zero_local_weight_and_nonunit_chunk_mean_are_preserved(weights):
    logprobs = torch.zeros(1, 2, 1, requires_grad=True)
    weight_tensor = torch.tensor([weights], requires_grad=True)
    loss, _ = native_loss(
        logprobs, torch.zeros_like(logprobs), torch.ones(1, 1), weight_tensor
    )
    loss.backward()
    torch.testing.assert_close(loss.detach(), -weight_tensor.detach().mean())
    torch.testing.assert_close(logprobs.grad.squeeze(-1), -weight_tensor.detach())
    assert weight_tensor.grad is None


@pytest.mark.parametrize(
    "weights",
    [
        torch.full((1, 2), -1.0),
        torch.tensor([[float("nan"), 1.0]]),
        torch.tensor([[float("inf"), 1.0]]),
    ],
)
def test_invalid_weights_fail_before_loss(weights):
    with pytest.raises(ValueError, match="finite and non-negative"):
        native_loss(
            torch.zeros(1, 2, 1), torch.zeros(1, 2, 1), torch.ones(1, 1), weights
        )


def test_wrong_domain_and_double_application_are_rejected():
    kwargs = {
        "logprobs": torch.zeros(1, 2, 1),
        "old_logprobs": torch.zeros(1, 2, 1),
        "advantages": torch.ones(1, 1),
        "single_action_dim": 1,
        "reward_type": "chunk_level",
        "dvac_chunk_advantage_weights": torch.ones(1, 2),
    }
    with pytest.raises(ValueError, match="chunk logprobs"):
        preprocess_loss_inputs(**kwargs, logprob_type="action_level")
    with pytest.raises(ValueError, match="not both"):
        preprocess_loss_inputs(
            **kwargs,
            logprob_type="chunk_level",
            dvac_advantage_weights=torch.ones(1, 2),
        )
    kwargs["dvac_chunk_advantage_weights"] = torch.ones(1, 3)
    with pytest.raises(ValueError, match=r"\[B,H\]"):
        preprocess_loss_inputs(**kwargs, logprob_type="chunk_level")
