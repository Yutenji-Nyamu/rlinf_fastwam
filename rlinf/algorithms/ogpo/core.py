# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure PyTorch mathematics shared by the RoboTwin OGPO adapters.

This module ports the numerical semantics from ``OGPO_public`` commit
``0b3be413cde766a41257c6b19c0c2b06393a557f``.  It deliberately contains no
rollout, replay, or model-specific behavior.  At the chain-likelihood
boundary the model action is flattened: ``D = H_model * D_model``.
"""

import math

import torch
from torch import Tensor

_LOG_2_PI = math.log(2.0 * math.pi)


def _require_positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}")
    return value


def _require_probability(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value}")
    return value


def _require_broadcasts_to(value: Tensor, target: Tensor, name: str) -> None:
    try:
        broadcast_shape = torch.broadcast_shapes(value.shape, target.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"{name} shape {tuple(value.shape)} cannot broadcast to "
            f"{tuple(target.shape)}"
        ) from exc
    if broadcast_shape != target.shape:
        raise ValueError(
            f"{name} shape {tuple(value.shape)} must broadcast exactly to "
            f"{tuple(target.shape)}, got {tuple(broadcast_shape)}"
        )


def condot_score_from_velocity(
    velocity: Tensor,
    state: Tensor,
    time: Tensor,
) -> Tensor:
    """Recover the CondOT marginal score ``(t * v - x) / (1 - t)``.

    ``velocity`` and ``state`` have the same action shape. ``time`` is a
    scalar or has singleton action dimensions and broadcasts to that shape.
    The formula is singular at ``t=1``; the tapered correction below performs
    the analytic cancellation and should be used on the final flow interval.
    """
    if velocity.shape != state.shape:
        raise ValueError(
            "velocity and state must have the same shape, got "
            f"{tuple(velocity.shape)} and {tuple(state.shape)}"
        )
    _require_broadcasts_to(time, state, "time")
    return (time * velocity - state) / (1.0 - time)


def tapered_noise_std(time: Tensor, *, sigma_init: float) -> Tensor:
    """Return official tapered transition std ``sigma_init * sqrt(1 - t)``.

    ``time`` uses OGPO coordinates: zero is the initial noise endpoint and one
    is the data/action endpoint.  The released sampler uses this value directly
    as the discrete Gaussian transition standard deviation; it does not add a
    further ``sqrt(dt)`` factor.
    """
    sigma_init = _require_positive_finite(sigma_init, "sigma_init")
    return sigma_init * torch.sqrt(torch.clamp_min(1.0 - time, 0.0))


def tapered_sde_drift_correction(
    velocity: Tensor,
    state: Tensor,
    time: Tensor,
    *,
    sigma_init: float,
) -> Tensor:
    """Return the marginal-preserving tapered-SDE drift correction.

    With ``sigma_t = sigma_init * sqrt(1-t)``, the ``1-t`` denominator in the
    CondOT score cancels analytically:

    ``correction = 0.5 * sigma_init**2 * (t * velocity - state)``.

    The sampler/scorer adds this correction to the flow velocity before its
    Euler update ``mean_next = state + (velocity + correction) * dt``.
    """
    if velocity.shape != state.shape:
        raise ValueError(
            "velocity and state must have the same shape, got "
            f"{tuple(velocity.shape)} and {tuple(state.shape)}"
        )
    _require_broadcasts_to(time, state, "time")
    sigma_init = _require_positive_finite(sigma_init, "sigma_init")
    return 0.5 * sigma_init**2 * (time * velocity - state)


def tapered_sde_step_stats(
    state: Tensor,
    velocity: Tensor,
    time: Tensor,
    *,
    step_size: float,
    sigma_init: float,
) -> tuple[Tensor, Tensor]:
    """Return ``(mean_next, std_next)`` for one official tapered-SDE step.

    No intermediate or final action clipping is performed here. That belongs
    to the model adapter, while the raw mean and raw sampled chain remain the
    likelihood coordinates.
    """
    step_size = _require_positive_finite(step_size, "step_size")
    correction = tapered_sde_drift_correction(
        velocity,
        state,
        time,
        sigma_init=sigma_init,
    )
    mean_next = state + (velocity + correction) * step_size
    std_next = tapered_noise_std(time, sigma_init=sigma_init).expand_as(state)
    return mean_next, std_next


def diagonal_gaussian_log_prob(
    sample: Tensor,
    mean: Tensor,
    std: Tensor,
) -> Tensor:
    """Return elementwise log-density under a diagonal Gaussian.

    All tensors have the same shape. Reduction is intentionally left to the
    caller so the whole-chain helper can apply the official horizon and action
    dimension normalizations exactly once.
    """
    if sample.shape != mean.shape or sample.shape != std.shape:
        raise ValueError(
            "sample, mean, and std must have the same shape, got "
            f"{tuple(sample.shape)}, {tuple(mean.shape)}, and {tuple(std.shape)}"
        )
    if (
        not sample.is_floating_point()
        or not mean.is_floating_point()
        or not std.is_floating_point()
    ):
        raise TypeError("sample, mean, and std must be floating-point tensors")
    return (
        -0.5 * ((sample - mean) / std).square()
        - torch.log(std)
        - 0.5 * _LOG_2_PI
    )


def normalized_whole_chain_log_prob(
    initial_state: Tensor,
    transition_states: Tensor,
    transition_means: Tensor,
    transition_stds: Tensor,
    *,
    normalize_denoising_horizon: bool = True,
    normalize_act_space_dimension: bool = True,
) -> Tensor:
    """Score one complete raw denoising chain.

    Shapes are:

    - ``initial_state``: ``[..., D]``;
    - transition tensors: ``[..., K, D]``;
    - return value: ``[...]``.

    The initial factor is ``N(0, I)`` and each of the ``K`` transition factors
    is the provided diagonal Gaussian.  With both normalizations enabled, the
    sum is divided by ``(K + 1) * D``.  For pi0 K4/H50/D32 this is ``5 * 1600``.
    Raw latent chain states belong here; projected/clipped environment actions
    do not.
    """
    if initial_state.ndim < 1:
        raise ValueError("initial_state must have at least one action dimension")
    if transition_states.ndim != initial_state.ndim + 1:
        raise ValueError(
            "transition_states must insert one K axis before the flattened "
            f"action axis; got {tuple(initial_state.shape)} and "
            f"{tuple(transition_states.shape)}"
        )
    if (
        transition_states.shape != transition_means.shape
        or transition_states.shape != transition_stds.shape
    ):
        raise ValueError(
            "transition states, means, and stds must have the same shape"
        )
    if transition_states.shape[:-2] != initial_state.shape[:-1]:
        raise ValueError("initial and transition leading dimensions must match")
    if transition_states.shape[-1] != initial_state.shape[-1]:
        raise ValueError(
            "initial and transition flattened action dimensions must match"
        )

    flow_steps = transition_states.shape[-2]
    action_dimension = initial_state.shape[-1]
    if flow_steps <= 0 or action_dimension <= 0:
        raise ValueError("whole-chain score requires K > 0 and D > 0")

    initial_log_prob = diagonal_gaussian_log_prob(
        initial_state,
        torch.zeros_like(initial_state),
        torch.ones_like(initial_state),
    ).sum(dim=-1)
    transition_log_prob = diagonal_gaussian_log_prob(
        transition_states,
        transition_means,
        transition_stds,
    ).sum(dim=-1).sum(dim=-1)
    score = initial_log_prob + transition_log_prob

    if normalize_denoising_horizon:
        score = score / (flow_steps + 1)
    if normalize_act_space_dimension:
        score = score / action_dimension
    return score


@torch.no_grad()
def conservative_group_advantages(q_values: Tensor) -> Tensor:
    """Compute OGPO+CA advantages from ``q_values[B, G, M]``.

    A per-head baseline is first taken over the ``G`` imagined candidates.
    If all ``M`` heads agree that a candidate advantage is positive, return
    the smallest positive value. If all agree it is negative, return the
    largest (least-magnitude) negative value. Otherwise return zero.
    """
    if q_values.ndim != 3:
        raise ValueError(
            f"q_values must have shape [B, G, M], got {tuple(q_values.shape)}"
        )
    if min(q_values.shape) <= 0:
        raise ValueError("B, G, and M must all be positive")
    if not q_values.is_floating_point():
        raise TypeError("q_values must be floating point")

    per_head = q_values - q_values.mean(dim=1, keepdim=True)
    minimum = per_head.min(dim=2).values
    maximum = per_head.max(dim=2).values
    return torch.where(
        minimum > 0.0,
        minimum,
        torch.where(maximum < 0.0, maximum, torch.zeros_like(minimum)),
    )


def clipped_ppo_loss(
    current_log_prob: Tensor,
    old_log_prob: Tensor,
    advantages: Tensor,
    *,
    clip_epsilon: float,
) -> Tensor:
    """Return the scalar OGPO clipped surrogate loss.

    Inputs normally have shape ``[B, G]`` and contain scalar normalized
    whole-chain scores and fixed conservative advantages. Only
    ``current_log_prob`` remains on the actor gradient path.
    """
    if (
        current_log_prob.shape != old_log_prob.shape
        or current_log_prob.shape != advantages.shape
    ):
        raise ValueError(
            "current_log_prob, old_log_prob, and advantages must have the same shape"
        )
    if current_log_prob.numel() == 0:
        raise ValueError("PPO loss requires at least one sample")
    if not all(
        tensor.is_floating_point()
        for tensor in (current_log_prob, old_log_prob, advantages)
    ):
        raise TypeError("PPO log probabilities and advantages must be floating point")
    clip_epsilon = _require_probability(clip_epsilon, "clip_epsilon")

    ratio = torch.exp(current_log_prob - old_log_prob.detach())
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    fixed_advantages = advantages.detach()
    surrogate = torch.minimum(
        ratio * fixed_advantages,
        clipped_ratio * fixed_advantages,
    )
    return -surrogate.mean()


@torch.no_grad()
def h_step_td_target(
    rewards: Tensor,
    bootstrap_mask: Tensor,
    next_target_q: Tensor,
    *,
    gamma: float,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Build a primitive-time h-step TD target.

    ``rewards`` has shape ``[..., H]``; ``bootstrap_mask`` and
    ``next_target_q`` have shape ``[...]``. ``valid_mask`` may mark a shorter
    prefix for each row and must have the same shape as ``rewards``. The result
    is

    ``sum_i gamma**i * r_i + gamma**h * mask * Q_target(s_h, a_next)``.

    Thus ``gamma`` is always one primitive waypoint and is never macro-rooted.
    ``next_target_q`` is already aggregated over the target-Q ensemble.
    """
    if rewards.ndim < 1 or rewards.shape[-1] <= 0:
        raise ValueError("rewards must have shape [..., H] with H > 0")
    if not rewards.is_floating_point():
        raise TypeError("rewards must be floating point")
    expected_shape = rewards.shape[:-1]
    if (
        bootstrap_mask.shape != expected_shape
        or next_target_q.shape != expected_shape
    ):
        raise ValueError(
            "bootstrap_mask and next_target_q must match rewards leading shape "
            f"{tuple(expected_shape)}"
        )
    gamma = _require_probability(gamma, "gamma")

    horizon = rewards.shape[-1]
    powers = torch.arange(horizon, dtype=rewards.dtype, device=rewards.device)
    discounts = torch.as_tensor(
        gamma,
        dtype=rewards.dtype,
        device=rewards.device,
    ).pow(powers)

    if valid_mask is None:
        selected_rewards = rewards
        bootstrap_discount: Tensor | float = gamma**horizon
    else:
        if valid_mask.shape != rewards.shape or valid_mask.dtype != torch.bool:
            raise ValueError(
                "valid_mask must be boolean and have the same shape as rewards"
            )
        selected_rewards = torch.where(
            valid_mask,
            rewards,
            torch.zeros_like(rewards),
        )
        effective_horizon = valid_mask.sum(dim=-1)
        bootstrap_discount = torch.as_tensor(
            gamma,
            dtype=rewards.dtype,
            device=rewards.device,
        ).pow(effective_horizon)

    return (
        (selected_rewards * discounts).sum(dim=-1)
        + bootstrap_discount
        * bootstrap_mask.to(dtype=rewards.dtype)
        * next_target_q.to(dtype=rewards.dtype)
    ).detach()
