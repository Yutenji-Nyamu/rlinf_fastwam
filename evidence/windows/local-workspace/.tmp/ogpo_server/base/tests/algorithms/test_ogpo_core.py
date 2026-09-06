import math

import pytest
import torch

from rlinf.algorithms.ogpo.core import (
    clipped_ppo_loss,
    condot_score_from_velocity,
    conservative_group_advantages,
    diagonal_gaussian_log_prob,
    h_step_td_target,
    normalized_whole_chain_log_prob,
    tapered_noise_std,
    tapered_sde_drift_correction,
    tapered_sde_step_stats,
)


def test_tapered_schedule_score_and_cancelled_correction() -> None:
    time = torch.tensor([[0.0], [0.75], [1.0]], dtype=torch.float64)
    sigma = tapered_noise_std(time, sigma_init=2.0)
    torch.testing.assert_close(
        sigma,
        torch.tensor([[2.0], [1.0], [0.0]], dtype=torch.float64),
    )

    state = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    velocity = torch.tensor([[4.0, 2.0]], dtype=torch.float64)
    half_time = torch.tensor([[0.5]], dtype=torch.float64)
    torch.testing.assert_close(
        condot_score_from_velocity(velocity, state, half_time),
        torch.tensor([[2.0, -2.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        tapered_sde_drift_correction(
            velocity,
            state,
            half_time,
            sigma_init=0.1,
        ),
        torch.tensor([[0.005, -0.005]], dtype=torch.float64),
    )
    mean_next, std_next = tapered_sde_step_stats(
        state,
        velocity,
        half_time,
        step_size=0.25,
        sigma_init=0.1,
    )
    torch.testing.assert_close(
        mean_next,
        torch.tensor([[2.00125, 2.49875]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        std_next,
        torch.full_like(state, 0.1 * math.sqrt(0.5)),
    )


def test_gaussian_and_normalized_whole_chain_score() -> None:
    sample = torch.tensor([0.0, 1.0], dtype=torch.float64)
    mean = torch.zeros_like(sample)
    std = torch.ones_like(sample)
    expected = torch.tensor(
        [-0.5 * math.log(2.0 * math.pi), -0.5 - 0.5 * math.log(2.0 * math.pi)],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        diagonal_gaussian_log_prob(sample, mean, std),
        expected,
    )

    # Three N(0, I) factors (initial + K=2) over D=2. Dividing by 3*2
    # therefore recovers the scalar standard-normal log-density at zero.
    initial = torch.zeros(1, 2, dtype=torch.float64)
    transitions = torch.zeros(1, 2, 2, dtype=torch.float64)
    score = normalized_whole_chain_log_prob(
        initial,
        transitions,
        torch.zeros_like(transitions),
        torch.ones_like(transitions),
    )
    torch.testing.assert_close(
        score,
        torch.tensor([-0.5 * math.log(2.0 * math.pi)], dtype=torch.float64),
    )

    # The identical raw chain under identical target/online means has ratio 1.
    current = normalized_whole_chain_log_prob(
        initial,
        transitions,
        torch.zeros_like(transitions),
        torch.ones_like(transitions),
    )
    torch.testing.assert_close(torch.exp(current - score), torch.ones_like(score))


def test_conservative_advantage_requires_all_head_signs_to_agree() -> None:
    # Each Q-head column has zero group mean, so these are also the per-head
    # advantages. Candidate 0 is unanimously positive, candidate 1 disagrees,
    # and candidate 2 is unanimously negative.
    q_values = torch.tensor(
        [[[3.0, 2.0, 4.0], [0.0, 1.0, -1.0], [-3.0, -3.0, -3.0]]]
    )
    advantages = conservative_group_advantages(q_values)
    torch.testing.assert_close(advantages, torch.tensor([[2.0, 0.0, -3.0]]))


def test_clipped_ppo_loss_matches_hand_computation_and_keeps_actor_grad() -> None:
    current = torch.log(torch.tensor([[1.2, 0.8, 1.0]])).requires_grad_(True)
    old = torch.zeros_like(current)
    advantages = torch.tensor([[1.0, -1.0, 2.0]])
    loss = clipped_ppo_loss(
        current,
        old,
        advantages,
        clip_epsilon=0.1,
    )
    # min(1.2, 1.1)*1 + min(-0.8, -0.9) + 1*2 = 2.2; loss=-2.2/3.
    torch.testing.assert_close(loss, torch.tensor(-2.2 / 3.0))
    loss.backward()
    assert current.grad is not None
    assert current.grad[0, 2].abs() > 0


def test_h_step_td_target_uses_primitive_gamma_and_variable_prefix() -> None:
    rewards = torch.tensor([[1.0, 2.0, 4.0], [1.0, 2.0, 99.0]])
    valid = torch.tensor([[True, True, True], [True, True, False]])
    target = h_step_td_target(
        rewards,
        bootstrap_mask=torch.tensor([1.0, 1.0]),
        next_target_q=torch.tensor([8.0, 8.0]),
        gamma=0.5,
        valid_mask=valid,
    )
    # Row 0: 1 + .5*2 + .25*4 + .5^3*8 = 4.
    # Row 1: 1 + .5*2 + .5^2*8 = 4; padded reward is ignored.
    torch.testing.assert_close(target, torch.tensor([4.0, 4.0]))

    terminal = h_step_td_target(
        rewards[1:, :2],
        bootstrap_mask=torch.tensor([0.0]),
        next_target_q=torch.tensor([123.0]),
        gamma=0.5,
    )
    torch.testing.assert_close(terminal, torch.tensor([2.0]))


def test_core_shape_errors_are_local_and_clear() -> None:
    with pytest.raises(ValueError, match=r"\[B, G, M\]"):
        conservative_group_advantages(torch.zeros(2, 3))
    with pytest.raises(ValueError, match="same shape"):
        diagonal_gaussian_log_prob(
            torch.zeros(2),
            torch.zeros(1),
            torch.ones(2),
        )
