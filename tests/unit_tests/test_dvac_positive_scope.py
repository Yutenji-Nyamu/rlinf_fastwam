import pytest
import torch

from rlinf.algorithms.dvac_train_weighting import (
    scope_dvac_weights,
    straight_through_scale_logprobs,
)
from rlinf.algorithms.losses import compute_ppo_actor_loss
from rlinf.algorithms.utils import preprocess_loss_inputs


@pytest.mark.parametrize("time_dim", [False, True])
def test_scope_uses_original_advantages_and_preserves_negative_feedback(time_dim):
    w = torch.tensor([[0.5, 1.5], [0.5, 1.5], [0.5, 1.5]], requires_grad=True)
    a = torch.tensor([[2.0], [-1.0], [0.0]], requires_grad=True)
    if time_dim:
        w, a = w.unsqueeze(0), a.unsqueeze(0)
    assert scope_dvac_weights(w, a) is w
    result = scope_dvac_weights(w, a, "positive")
    expected = torch.tensor([[0.5, 1.5], [1.0, 1.0], [1.0, 1.0]])
    assert torch.equal(result, expected.unsqueeze(0) if time_dim else expected)
    assert not result.requires_grad


def test_invalid_scope_or_shape_fails_explicitly():
    with pytest.raises(ValueError, match="advantage_scope"):
        scope_dvac_weights(torch.ones(2, 50), torch.ones(2, 1), "bad")
    with pytest.raises(ValueError, match="one original advantage"):
        scope_dvac_weights(torch.ones(2, 50), torch.ones(2, 50), "positive")


def native_chunk_loss(lp, old, a):
    inputs = preprocess_loss_inputs(
        logprobs=lp,
        old_logprobs=old,
        advantages=a,
        logprob_type="chunk_level",
        reward_type="chunk_level",
        single_action_dim=2,
    )
    return compute_ppo_actor_loss(
        logprobs=inputs["logprobs"],
        old_logprobs=inputs["old_logprobs"],
        advantages=inputs["advantages"],
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        loss_mask=torch.ones(len(lp), 1, dtype=torch.bool),
        loss_mask_sum=torch.ones(len(lp), 1),
        max_episode_steps=1,
    )


@pytest.mark.parametrize("all_ones", [False, True])
def test_native_chunk_clip_forward_identical_and_correct_scoped_gradients(all_ones):
    # Includes unclipped and clipped positive/negative cases plus A=0.
    ratios = torch.tensor([1.05, 1.645, 0.7, 1.05, 0.7, 1.645, 1.0])
    a = torch.tensor([[1.0], [1.0], [1.0], [-1.0], [-1.0], [-1.0], [0.0]])
    lp = (ratios.log()[:, None, None] / 100).expand(-1, 50, 2).clone().requires_grad_()
    reference = lp.detach().clone().requires_grad_()
    w = torch.linspace(0.5, 1.5, 50).expand(len(lp), -1).clone()
    if all_ones:
        w.fill_(1)
    effective = scope_dvac_weights(w, a, "positive")
    transformed = straight_through_scale_logprobs(lp, effective)
    assert torch.equal(transformed, reference)
    loss, metrics = native_chunk_loss(transformed, torch.zeros_like(lp), a)
    control_loss, control_metrics = native_chunk_loss(reference, torch.zeros_like(lp), a)
    assert torch.equal(loss, control_loss)
    for key in metrics:
        torch.testing.assert_close(metrics[key], control_metrics[key], equal_nan=True)
    loss.backward()
    control_loss.backward()
    torch.testing.assert_close(lp.grad, reference.grad * effective.unsqueeze(-1))
    assert torch.equal(lp.grad[a[:, 0] <= 0], reference.grad[a[:, 0] <= 0])
    assert torch.count_nonzero(lp.grad[1]) == 0  # chunk 1.645 clips positive A
    assert torch.count_nonzero(lp.grad[3]) > 0  # negative feedback remains


def test_all_scope_preserves_existing_both_sides_weighting():
    a = torch.tensor([[1.0], [-1.0]])
    w = torch.tensor([[0.5, 1.5], [0.5, 1.5]])
    lp = torch.ones(2, 2, 1, requires_grad=True)
    straight_through_scale_logprobs(lp, scope_dvac_weights(w, a, "all")).sum().backward()
    assert torch.equal(lp.grad.squeeze(-1), w)
