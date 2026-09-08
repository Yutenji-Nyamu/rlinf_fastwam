"""Focused CPU checks; execute only in the source-locked server environment."""

import pytest
import torch

from rlinf.algorithms.advantages import compute_prism_rloo_advantages
from rlinf.algorithms.dvac_rank_reward import (
    reverse_rank_quality,
    trajectory_dvac_quality,
    trajectory_mean_log_variance,
)
from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.utils.metric_utils import compute_loss_mask


def test_executed_mask_ignores_unexecuted_tail_and_detaches_quality():
    log_values = torch.tensor([[[1.0, 1.0], [1.0, 20.0], [3.0, 20.0], [5.0, 20.0]]])
    variance = log_values.exp().requires_grad_()
    mask = torch.tensor([[[True, True], [True, False], [True, False], [True, False]]])
    cost, quality = trajectory_dvac_quality(variance, mask, group_size=4, log_eps=1e-12)
    torch.testing.assert_close(cost, torch.tensor([1.0, 1.0, 3.0, 5.0]))
    torch.testing.assert_close(quality, torch.tensor([5 / 6, 5 / 6, 1 / 3, 0.0]))
    assert not cost.requires_grad and not quality.requires_grad


def test_rank_keeps_group_identity_and_ties():
    cost = torch.tensor([0.0, 1.0, 2.0, 3.0, 100.0, 100.0, 100.0, 100.0])
    quality = reverse_rank_quality(cost, 4).reshape(2, 4)
    torch.testing.assert_close(quality[0], torch.tensor([1.0, 2 / 3, 1 / 3, 0.0]))
    torch.testing.assert_close(quality[1], torch.full((4,), 0.5))
    torch.testing.assert_close(quality.mean(-1), torch.full((2,), 0.5))


@pytest.mark.parametrize("outcome", [0.0, 1.0])
def test_same_outcome_groups_receive_bounded_rloo_without_std(outcome):
    quality = torch.arange(8, dtype=torch.float32) / 7
    rewards = torch.full((1, 8), outcome)
    mask = torch.ones(2, 8, dtype=torch.bool)
    adv, returns = compute_prism_rloo_advantages(rewards, mask, 8, quality, 0.2)
    expected = 0.2 * 8 / 7 * (quality - 0.5)
    torch.testing.assert_close(adv, expected.expand(2, -1), atol=1e-7, rtol=1e-6)
    assert returns is None
    assert adv.max().item() < 0.115 and adv.min().item() > -0.115
    zero, _ = compute_prism_rloo_advantages(rewards, mask, 8, torch.full((8,), 0.5), 0.2)
    torch.testing.assert_close(zero, torch.zeros_like(zero), atol=1e-7, rtol=0)


def test_mixed_group_uses_quality_and_preserves_success_priority():
    quality = torch.arange(8, dtype=torch.float32) / 7
    rewards = torch.tensor([[1.0, 0, 0, 0, 0, 0, 0, 0]])
    adv, _ = compute_prism_rloo_advantages(rewards, torch.ones(1, 8), 8, quality, 0.2)
    combined = rewards + 0.2 * quality
    expected = (combined * 8 - combined.sum(-1, keepdim=True)) / 7
    torch.testing.assert_close(adv, expected)
    assert combined[0, 0] > combined[0, 1:].max()
    assert adv[0, 0] > 0 and (adv[0, 1:] < 0).all()


def test_registry_uses_original_rewards_and_terminal_mask_with_chunk_loss():
    # Two chunks, four trajectories in one scene; success can happen in either chunk.
    rewards = torch.zeros(2, 4, 2)
    rewards[0, 0, 0] = 1
    rewards[1, 1, 1] = 1
    original_rewards = rewards.clone()
    dones = torch.zeros(3, 4, 2, dtype=torch.bool)
    dones[1:, 0] = True
    action_mask, mask_sum = compute_loss_mask(dones)
    quality = torch.tensor([0.0, 1.0, 2 / 3, 1 / 3])
    result = calculate_adv_and_returns(
        task_type="embodied", adv_type="prism_rloo", rewards=rewards, dones=dones,
        loss_mask=action_mask.any(-1, keepdim=True), loss_mask_sum=mask_sum[..., -1:],
        group_size=4, reward_type="chunk_level", trajectory_quality=quality,
        quality_lambda=0.2,
    )
    adv = result["advantages"]
    assert adv.shape == (2, 4, 1) and adv[1, 0, 0] == 0
    assert adv[0, 0, 0] > 0 and adv[0, 1, 0] > 0
    assert (adv[:, 2:] < 0).all()
    torch.testing.assert_close(rewards, original_rewards)
    logprobs = torch.zeros(8, 2, 14, requires_grad=True)
    flat_mask = action_mask.any(-1).reshape(8)
    loss, _ = policy_loss(
        task_type="embodied", loss_type="actor", logprob_type="chunk_level",
        reward_type="chunk_level", single_action_dim=14, logprobs=logprobs,
        old_logprobs=torch.zeros_like(logprobs), advantages=adv.reshape(8, 1),
        loss_mask=flat_mask, clip_ratio_low=0.2, clip_ratio_high=0.2,
    )
    loss.backward()
    assert torch.isfinite(logprobs.grad).all()
    assert logprobs.grad[4].abs().sum() == 0
    assert logprobs.grad[0].sum() < 0 and logprobs.grad[2].sum() > 0


@pytest.mark.parametrize("case", ["empty_trajectory", "nonfinite_variance", "negative_variance"])
def test_quality_rejects_invalid_semantic_inputs(case):
    variance = torch.ones(1, 4, 2)
    mask = torch.ones_like(variance, dtype=torch.bool)
    if case == "empty_trajectory":
        mask[:, 0] = False
    elif case == "nonfinite_variance":
        variance[0, 0, 0] = float("nan")
    else:
        variance[0, 0, 0] = -1
    with pytest.raises(ValueError):
        trajectory_mean_log_variance(variance, mask, log_eps=1e-12)


def test_prism_rejects_nonbinary_episode_reward():
    with pytest.raises(ValueError, match="binary"):
        compute_prism_rloo_advantages(
            torch.full((1, 4), 0.5), torch.ones(1, 4), 4, torch.linspace(0, 1, 4), 0.2
        )
