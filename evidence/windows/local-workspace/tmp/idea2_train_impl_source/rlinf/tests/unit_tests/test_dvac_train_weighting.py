from __future__ import annotations

import torch
import pytest

from rlinf.algorithms.dvac_train_weighting import (
    DVACRecentStats,
    DVACStepStats,
    compute_endpoint_variances,
    local_log_v_sufficient_statistics,
    straight_through_scale_logprobs,
)


def test_endpoint_variance_uses_population_tail_variance() -> None:
    z = torch.tensor(
        [[[[0.0]], [[1.0]], [[3.0]], [[7.0]]]], dtype=torch.float32
    )
    result = compute_endpoint_variances(z, (2, 3, 4))
    assert torch.equal(result[2], torch.tensor([[4.0]]))
    assert torch.allclose(result[3], torch.tensor([[56.0 / 9.0]]))
    assert torch.allclose(result[4], torch.tensor([[7.1875]]))


def test_recent_stats_warmup_apply_and_window_eviction() -> None:
    recent = DVACRecentStats(
        window_steps=2,
        warmup_steps=1,
        log_eps=1e-12,
        std_floor=1e-6,
        z_clip=2.0,
        strength=0.1,
    )
    variance = torch.exp(torch.tensor([[[0.0, 1.0, 4.0]]]))
    weights, clipped_z, warmup, history = recent.compute_weights(variance)
    assert warmup
    assert history["history_steps"] == 0
    assert torch.equal(weights, torch.ones_like(weights))
    assert torch.equal(clipped_z, torch.zeros_like(clipped_z))

    recent.push(DVACStepStats(0, 2, 1.0, 1.0))  # mean=.5, std=.5
    weights, clipped_z, warmup, history = recent.compute_weights(variance)
    assert not warmup
    assert history["history_steps"] == 1
    assert torch.allclose(clipped_z, torch.tensor([[[-1.0, 1.0, 2.0]]]))
    assert torch.allclose(weights, torch.tensor([[[0.9, 1.1, 1.2]]]))

    recent.push(DVACStepStats(1, 1, 2.0, 4.0))
    recent.push(DVACStepStats(2, 1, 3.0, 9.0))
    state = recent.state_dict()
    assert [item["runner_step"] for item in state["steps"]] == [1, 2]


def test_weight_range_cannot_reverse_advantage_direction() -> None:
    with pytest.raises(ValueError, match="cannot reverse"):
        DVACRecentStats(
            window_steps=5,
            warmup_steps=1,
            log_eps=1e-12,
            std_floor=1e-6,
            z_clip=2.0,
            strength=0.6,
        )


def test_local_stats_respect_query_loss_mask() -> None:
    variance = torch.exp(
        torch.tensor([[[0.0, 1.0], [10.0, 20.0]]], dtype=torch.float64)
    )
    mask = torch.tensor([[[True], [False]]])
    stats = local_log_v_sufficient_statistics(variance, mask, log_eps=1e-12)
    assert torch.allclose(stats, torch.tensor([2.0, 1.0, 1.0], dtype=torch.float64))


def test_straight_through_keeps_forward_and_scales_per_h_gradient() -> None:
    logprobs = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],
        dtype=torch.float16,
        requires_grad=True,
    )
    weights = torch.tensor([[0.8, 1.0, 1.2]])
    scaled = straight_through_scale_logprobs(logprobs, weights)
    assert scaled.dtype == logprobs.dtype
    assert torch.equal(scaled, logprobs)
    scaled.sum().backward()
    expected = weights.to(logprobs).unsqueeze(-1).expand_as(logprobs)
    assert torch.allclose(logprobs.grad, expected)


def test_trajectory_shuffle_keeps_query_signal_and_metadata_aligned() -> None:
    from rlinf.workers.actor.fsdp_actor_worker import process_nested_dict_for_train

    num_steps, batch_size, horizon = 2, 3, 4
    query_id = torch.arange(num_steps * batch_size).reshape(num_steps, batch_size)
    weights = query_id.unsqueeze(-1).expand(-1, -1, horizon).float()
    batch = {
        "prev_logprobs": weights.unsqueeze(-1),
        "dones": torch.arange((num_steps + 1) * batch_size).reshape(
            num_steps + 1, batch_size
        ),
        "forward_inputs": {
            "dvac_weights": weights,
            "dvac_meta_query_id": query_id,
        },
    }
    shuffle = torch.tensor([4, 0, 5, 2, 1, 3])
    result = process_nested_dict_for_train(batch, shuffle)
    expected_id = query_id.reshape(-1)[shuffle]
    assert torch.equal(result["forward_inputs"]["dvac_meta_query_id"], expected_id)
    assert torch.equal(result["forward_inputs"]["dvac_weights"][:, 0], expected_id)
    assert result["dones"].shape[0] == num_steps * batch_size
