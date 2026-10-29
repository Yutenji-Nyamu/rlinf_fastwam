import pytest
import torch

from rlinf.algorithms.pi05_distillation.fql import actor_objective
from rlinf.algorithms.pi05_distillation.q_evaluation import pooled_features, summarize


def test_observed_returns_stop_once_and_do_not_bootstrap_at_limit():
    rows = [
        dict(slot=0, query=0, q1=0.8, q2=1.0, reward=0, terminal=False),
        dict(slot=1, query=0, q1=2.0, q2=3.0, reward=0, terminal=True),
        dict(slot=0, query=1, q1=0.8, q2=1.0, reward=1, terminal=True),
    ]
    output, summary = summarize(rows, 0.99)
    assert [r["actual_discounted_return"] for r in output] == [0.99, 0.0, 1.0]
    assert summary["episodes"] == 2 and summary["records"] == 3
    assert summary["outside_0_1_fraction"] == pytest.approx(1 / 3)
    assert summary["mae"] == pytest.approx((0.09 + 2.5 + 0.1) / 3)
    with pytest.raises(AssertionError, match="incomplete"):
        summarize(rows[:1], 0.99)


def test_masked_prefix_pool_matches_training_formula():
    prefix = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    valid = torch.tensor([[True, True, False]])
    state = torch.arange(32.0)[None]
    features = pooled_features(prefix, valid, state)
    torch.testing.assert_close(
        features, torch.cat((torch.tensor([[2.0, 3.0]]), state[:, :14]), -1)
    )


def test_alpha_thirty_changes_anchor_only():
    prediction = torch.ones(2, 50, 32, requires_grad=True)
    target = torch.zeros_like(prediction)
    q = torch.ones(2, 2, requires_grad=True)
    old, _ = actor_objective(prediction, target, q, 10)
    new, _ = actor_objective(prediction, target, q, 30)
    torch.testing.assert_close(new - old, torch.tensor(20.0))
    new.backward()
    assert not prediction.grad[..., 14:].any()
    torch.testing.assert_close(q.grad, torch.full_like(q, -0.25))
