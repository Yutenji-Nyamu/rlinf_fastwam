"""CPU tests of exact selection, native chunk-PPO gradients, and method RNG."""

import json
import random

import pytest
import torch

from rlinf.algorithms.dvac_top20 import (
    DVACTop20Config,
    DVACTop20State,
    compute_top20_weights,
    distributed_top20_weights,
)
from rlinf.algorithms.dvac_train_weighting import straight_through_scale_logprobs
from rlinf.algorithms.losses import compute_ppo_actor_loss
from rlinf.algorithms.utils import preprocess_loss_inputs


@pytest.mark.parametrize("domain", ["chunk", "batch"])
def test_exact_top_count_native_mask_and_mean(domain):
    v = torch.arange(4 * 50, dtype=torch.float32).reshape(4, 50)
    mask = torch.tensor([[True], [True], [False], [True]])
    w, metrics = compute_top20_weights(v, mask, torch.arange(4), selection_domain=domain)
    assert (w > 0).sum() == 30
    assert not w[2].any()
    assert torch.allclose(w[mask.expand_as(w)].mean(), torch.tensor(1.0))
    assert set(w.unique().tolist()) == {0.0, 5.0}
    if domain == "chunk":
        assert torch.equal((w > 0).sum(1), torch.tensor([10, 10, 0, 10]))
    else:
        # Not a per-rank/per-chunk substitute: the high-valued last chunk wins.
        assert torch.equal((w > 0).sum(1), torch.tensor([0, 0, 0, 30]))
    assert metrics["selected_fraction"] == 0.2


@pytest.mark.parametrize("domain", ["chunk", "batch"])
def test_ties_exact_count_and_reordering(domain):
    values = torch.ones(7, 50)
    ids = torch.tensor([901, 82, 42, 531, 67, 102, 453])
    order = torch.tensor([5, 4, 3, 2, 1, 0, 6])
    weights, _ = compute_top20_weights(values, None, ids, selection_domain=domain)
    reordered, _ = compute_top20_weights(values[order], None, ids[order], selection_domain=domain)
    assert torch.equal(weights[order], reordered)
    assert (weights > 0).sum() == 70
    assert not bool((weights[:, :10] > 0).all())


def test_nonfinite_invalid_and_small_domains():
    v = torch.tensor([[float("nan"), float("inf"), -3.0], [1.0, 2.0, 3.0]])
    mask = torch.tensor([[False], [True]])
    weights, _ = compute_top20_weights(v, mask, torch.arange(2))
    assert torch.equal(weights, torch.tensor([[0., 0., 0.], [0., 0., 3.]]))
    with pytest.raises(ValueError, match="finite"):
        compute_top20_weights(v, None, torch.arange(2))
    weights, stats = compute_top20_weights(v, torch.zeros(2, 1, dtype=torch.bool), torch.arange(2))
    assert not weights.any() and stats["valid_positions"] == 0
    with pytest.raises(ValueError, match="unique"):
        compute_top20_weights(torch.ones(2, 3), None, torch.zeros(2, dtype=torch.int64))


@pytest.mark.parametrize("values", [
    {"selection_domain": "microbatch"}, {"top_fraction": 0},
    {"top_fraction": float("nan")}, {"top_fraction": 1.1},
    {"full_update_probability": -0.1}, {"full_update_probability": 1.1},
    {"selected_l": 1}, {"selected_l": 3.5}, {"method_seed": "42"},
    {"enabled": "false"}, {"soft_mix": .5},
    {"top_fraction": True}, {"top_fraction": ".2"},
    {"full_update_probability": False}, {"full_update_probability": ".1"},
])
def test_invalid_config(values):
    with pytest.raises((TypeError, ValueError)):
        DVACTop20Config.from_dict(values)


def _native_loss(logprobs, advantages, mask):
    processed = preprocess_loss_inputs(
        logprobs=logprobs, old_logprobs=torch.zeros_like(logprobs),
        advantages=advantages, logprob_type="chunk_level", reward_type="chunk_level",
        single_action_dim=14, loss_mask=mask,
        loss_mask_sum=torch.tensor([[50], [100], [150], [200], [100], [150], [50], [200]]),
    )
    return compute_ppo_actor_loss(
        **processed, clip_ratio_low=.2, clip_ratio_high=.2, max_episode_steps=200,
    )


@pytest.mark.parametrize("q", [.2, 1.0])
@pytest.mark.parametrize("domain", ["chunk", "batch"])
def test_native_chunk_ratio_clip_and_direct_pg_gate(q, domain):
    ratios = torch.tensor([1.05, 1.4, .7, 1.05, .7, 4., 1.1, 1.05])
    a = torch.tensor([[1.], [1.], [1.], [-1.], [-1.], [-1.], [0.], [1.]])
    mask = torch.ones(8, 1, dtype=torch.bool)
    mask[-1] = False
    current = (ratios.log()[:, None, None] / 700).expand(-1, 50, 14).clone().requires_grad_()
    clean = current.detach().clone().requires_grad_()
    v = torch.arange(400).reshape(8, 50).float()
    weights, _ = compute_top20_weights(v, mask, torch.arange(8), top_fraction=q, selection_domain=domain)
    gated = straight_through_scale_logprobs(current, weights)
    assert torch.equal(gated, current)
    loss, metrics = _native_loss(gated, a, mask)
    clean_loss, clean_metrics = _native_loss(clean, a, mask)
    assert torch.equal(loss, clean_loss)
    assert metrics.keys() == clean_metrics.keys()
    for key in metrics:
        torch.testing.assert_close(torch.as_tensor(metrics[key]), torch.as_tensor(clean_metrics[key]), equal_nan=True)
    loss.backward()
    clean_loss.backward()
    torch.testing.assert_close(current.grad, clean.grad * weights[..., None], rtol=2e-5, atol=1e-7)
    assert not current.grad[(weights == 0)[..., None].expand_as(current)].any()
    # Positive over-upper clip, negative under-lower clip, A0 and mask0 stay blocked.
    assert not current.grad[[1, 4, 6, 7]].any()


def test_p0_p1_rng_independence_and_json_resume():
    policy_rng = torch.get_rng_state().clone()
    python_rng = random.getstate()
    for p, expected in [(0., False), (1., True)]:
        state = DVACTop20State(DVACTop20Config.from_dict({"full_update_probability": p}))
        assert [state.next_full_update() for _ in range(7)] == [expected] * 7
        assert state.update_index == 7
    config = DVACTop20Config.from_dict({"enabled": True, "full_update_probability": .3})
    state = DVACTop20State(config)
    for _ in range(13):
        state.next_full_update()
    payload = json.loads(json.dumps(state.state_dict()))
    expected = [state.next_full_update() for _ in range(30)]
    restored = DVACTop20State(config)
    restored.load_state_dict(payload)
    assert [restored.next_full_update() for _ in range(30)] == expected
    assert torch.equal(policy_rng, torch.get_rng_state())
    assert python_rng == random.getstate()
    incompatible = DVACTop20State(DVACTop20Config.from_dict({"enabled": True, "selection_domain": "chunk"}))
    with pytest.raises(ValueError, match="identity"):
        incompatible.load_state_dict(payload)
    payload["update_index"] = -1
    with pytest.raises(ValueError, match="counters"):
        DVACTop20State(config).load_state_dict(payload)


def test_distributed_helper_without_process_group():
    config = DVACTop20Config.from_dict({"enabled": True})
    v = torch.arange(100).reshape(2, 50).float()
    expected, _ = compute_top20_weights(v, None, torch.arange(2))
    actual, _ = distributed_top20_weights(v, None, torch.arange(2), config)
    assert torch.equal(expected, actual)
