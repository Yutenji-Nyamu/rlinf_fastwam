"""Exercise the native preprocessing/loss, not a standalone surrogate formula."""
import pytest
import torch

from rlinf.algorithms.dvac_train_weighting import scope_dvac_weights
from rlinf.algorithms.losses import compute_ppo_actor_loss
from rlinf.algorithms.utils import preprocess_loss_inputs


def native(lp, old, a, w=None, mask=None, clamp=False):
    inputs = preprocess_loss_inputs(
        logprobs=lp, old_logprobs=old, advantages=a,
        logprob_type="chunk_level", reward_type="chunk_level",
        single_action_dim=lp.shape[-1], loss_mask=mask,
        loss_mask_sum=torch.full_like(a, 200.0) if mask is not None else None,
        dvac_chunk_advantage_weights=w,
    )
    return compute_ppo_actor_loss(
        **inputs, clip_ratio_low=0.2, clip_ratio_high=0.2,
        max_episode_steps=200 if mask is not None else None,
        clip_log_ratio_min=-0.25 if clamp else None,
        clip_log_ratio_max=0.25 if clamp else None,
    )


@pytest.mark.parametrize("horizon", [2, 50])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("clamp", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_native_control_values_clip_and_position_gradients(horizon, masked, clamp, weighted):
    r = torch.tensor([1.05, 1.645, 0.7, 1.05, 0.7, 1.645, 1.0, 1.1])
    a = torch.tensor([[1.], [1.], [1.], [-1.], [-1.], [-1.], [0.], [1.]])
    lp = (r.log()[:, None, None] / (horizon * 14)).expand(-1, horizon, 14).clone().requires_grad_()
    ref = lp.detach().clone().requires_grad_()
    mask = torch.tensor([[1], [1], [1], [1], [1], [1], [1], [0]], dtype=torch.bool) if masked else None
    raw_w = torch.linspace(0.5, 1.5, horizon).expand(len(lp), -1).clone()
    if not weighted:
        raw_w.fill_(1)
    w = scope_dvac_weights(raw_w, a, "positive")
    loss, metrics = native(lp, torch.zeros_like(lp), a, w, mask, clamp)
    control, control_metrics = native(ref, torch.zeros_like(ref), a, None, mask, clamp)
    torch.testing.assert_close(loss, control, atol=2e-6, rtol=2e-6)
    for k in metrics:
        torch.testing.assert_close(metrics[k], control_metrics[k], equal_nan=True, atol=2e-6, rtol=2e-6)
    loss.backward(); control.backward()
    torch.testing.assert_close(lp.grad, ref.grad * w.unsqueeze(-1), atol=2e-6, rtol=2e-6)
    assert torch.count_nonzero(lp.grad[1]) == 0  # whole-chunk 1.645 clips


@pytest.mark.parametrize("weights", [[0.5, 1.5], [1.5, 0.5], [1.3, 1.3], [0., 2.]])
def test_advantages_are_explicit_detached_and_not_chunk_normalized(weights):
    lp = torch.zeros(1, 2, 1, requires_grad=True)
    w = torch.tensor([weights], requires_grad=True)
    inputs = preprocess_loss_inputs(
        logprobs=lp, old_logprobs=torch.zeros_like(lp), advantages=torch.ones(1, 1),
        single_action_dim=1, logprob_type="chunk_level", reward_type="chunk_level",
        dvac_chunk_advantage_weights=w,
    )
    assert torch.equal(inputs["chunk_action_advantages"], w)
    loss, _ = compute_ppo_actor_loss(**inputs, clip_ratio_low=.2, clip_ratio_high=.2)
    loss.backward()
    torch.testing.assert_close(lp.grad.squeeze(-1), -w.detach())
    torch.testing.assert_close(loss.detach(), -w.detach().mean())
    assert w.grad is None


def test_shared_network_parameters_receive_the_correct_action_contributions():
    torch.manual_seed(4)
    x = torch.randn(5, 3)
    model = torch.nn.Linear(3, 50 * 14, bias=False)
    a = torch.tensor([[1.], [-1.], [1.], [0.], [-1.]])
    lp = model(x).reshape(5, 50, 14) * .0001
    old = lp.detach().clone()
    w = scope_dvac_weights(torch.rand(5, 50) + .5, a, "positive")
    loss, _ = native(lp, old, a, w)
    gradient = torch.autograd.grad(loss, model.weight, retain_graph=True)[0]
    expected = torch.autograd.grad(-(a * (lp * w[..., None]).sum((1, 2))[:, None]).mean(), model.weight)[0]
    torch.testing.assert_close(gradient, expected)


def test_reject_wrong_granularity_and_negative_weights():
    kwargs = dict(logprobs=torch.zeros(1, 2, 1), old_logprobs=torch.zeros(1, 2, 1),
                  advantages=torch.ones(1, 1), single_action_dim=1, reward_type="chunk_level")
    with pytest.raises(ValueError, match="chunk logprobs"):
        preprocess_loss_inputs(**kwargs, logprob_type="action_level", dvac_chunk_advantage_weights=torch.ones(1, 2))
    with pytest.raises(ValueError, match="non-negative"):
        preprocess_loss_inputs(**kwargs, logprob_type="chunk_level", dvac_chunk_advantage_weights=-torch.ones(1, 2))
