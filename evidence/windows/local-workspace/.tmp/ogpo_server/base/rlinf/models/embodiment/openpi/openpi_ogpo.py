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

"""Source-locked OGPO chain math for the pi0 action expert.

The functions are callback based so the main OpenPI model only needs to build
one prefix cache and expose an OGPO-oriented velocity callback.  They do not
own replay, critic, optimizer, or environment state.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from rlinf.models.embodiment.modules.ogpo_modules import (
    ogpo_time_to_pi0_time,
    pi0_velocity_to_ogpo,
    project_ogpo_action_views,
)

# Callback contract: x is [B, G, H, D], time_ogpo is [B, G], and the return
# must have the same shape as x.  A real OpenPI wrapper may flatten B*G around
# its suffix-expert call while reusing/expanding the prefix cache.
OGPOVelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class OGPOChainSample:
    """EMA-generated imagined chains and their two downstream action views."""

    raw_chains: torch.Tensor
    old_chain_score: torch.Tensor
    raw_final_action: torch.Tensor
    canonical_action: torch.Tensor


def openpi_velocity_as_ogpo(
    velocity_pi0_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_t: torch.Tensor,
    time_ogpo: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a pi0 velocity callback in OGPO time/orientation."""
    time_pi0 = ogpo_time_to_pi0_time(time_ogpo)
    velocity_pi0 = velocity_pi0_fn(x_t, time_pi0)
    if velocity_pi0.shape != x_t.shape:
        raise ValueError(
            "velocity_pi0_fn must preserve x_t shape; got "
            f"{tuple(velocity_pi0.shape)} for {tuple(x_t.shape)}"
        )
    return pi0_velocity_to_ogpo(velocity_pi0)


def _validate_sigma(sigma_init: float) -> None:
    if not math.isfinite(sigma_init) or sigma_init <= 0.0:
        raise ValueError(f"sigma_init must be finite and positive, got {sigma_init}")


def _normal_log_prob_sum(
    value: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Gaussian log-density summed over the full H*D model action."""
    value_f32 = value.to(dtype=torch.float32)
    mean_f32 = mean.to(device=value.device, dtype=torch.float32)
    std_f32 = std.to(device=value.device, dtype=torch.float32)
    log_prob = -0.5 * (
        ((value_f32 - mean_f32) / std_f32).square()
        + 2.0 * torch.log(std_f32)
        + math.log(2.0 * math.pi)
    )
    return log_prob.sum(dim=(-2, -1))


def tapered_ogpo_mean_and_sigma(
    x_t: torch.Tensor,
    velocity_ogpo: torch.Tensor,
    time_ogpo: torch.Tensor,
    *,
    flow_steps: int,
    sigma_init: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one marginal-preserving OGPO Euler mean and tapered std."""
    if flow_steps <= 0:
        raise ValueError(f"flow_steps must be positive, got {flow_steps}")
    _validate_sigma(sigma_init)
    if velocity_ogpo.shape != x_t.shape:
        raise ValueError(
            "velocity_ogpo must match x_t; got "
            f"{tuple(velocity_ogpo.shape)} and {tuple(x_t.shape)}"
        )
    expected_time_shape = x_t.shape[:-2]
    if time_ogpo.shape != expected_time_shape:
        raise ValueError(
            "time_ogpo must match x_t leading dimensions; got "
            f"{tuple(time_ogpo.shape)} and {tuple(expected_time_shape)}"
        )

    x_f32 = x_t.to(dtype=torch.float32)
    velocity_f32 = velocity_ogpo.to(dtype=torch.float32)
    time = time_ogpo.to(device=x_t.device, dtype=torch.float32)[..., None, None]
    correction = 0.5 * sigma_init**2 * (time * velocity_f32 - x_f32)
    mean = x_f32 + (velocity_f32 + correction) / float(flow_steps)
    sigma = sigma_init * torch.sqrt((1.0 - time).clamp_min(0.0))
    return mean, sigma


@torch.no_grad()
def sample_ogpo_chains(
    velocity_ema_fn: OGPOVelocityFn,
    *,
    initial_noise: torch.Tensor,
    flow_steps: int,
    sigma_init: float,
    executed_horizon: int,
    active_action_dim: int,
    transition_noise: torch.Tensor | None = None,
) -> OGPOChainSample:
    """Generate full tapered raw chains under the EMA action expert.

    ``initial_noise`` is ``[B,G,H,D]`` and optional ``transition_noise`` is
    ``[B,G,K,H,D]``.  Gaussian innovations are clamped to ``[-3,3]`` exactly
    as in the released OGPO sampler.  There is no intermediate mean clip and
    no final action bound is written into the raw chain.
    """
    if initial_noise.ndim != 4:
        raise ValueError(
            "initial_noise must have shape [B,G,H,D], got "
            f"{tuple(initial_noise.shape)}"
        )
    if flow_steps <= 0:
        raise ValueError(f"flow_steps must be positive, got {flow_steps}")
    _validate_sigma(sigma_init)

    batch_size, group_size, horizon, action_dim = initial_noise.shape
    expected_transition_shape = (
        batch_size,
        group_size,
        flow_steps,
        horizon,
        action_dim,
    )
    if (
        transition_noise is not None
        and transition_noise.shape != expected_transition_shape
    ):
        raise ValueError(
            "transition_noise must have shape [B,G,K,H,D], got "
            f"{tuple(transition_noise.shape)}; expected {expected_transition_shape}"
        )

    x_t = initial_noise.detach().to(dtype=torch.float32).clone()
    raw_steps = [x_t]
    log_prob = _normal_log_prob_sum(
        x_t,
        torch.zeros((), device=x_t.device, dtype=x_t.dtype),
        torch.ones((), device=x_t.device, dtype=x_t.dtype),
    )

    for index in range(flow_steps):
        time_ogpo = torch.full(
            (batch_size, group_size),
            index / float(flow_steps),
            device=x_t.device,
            dtype=torch.float32,
        )
        velocity = velocity_ema_fn(x_t, time_ogpo)
        mean, sigma = tapered_ogpo_mean_and_sigma(
            x_t,
            velocity,
            time_ogpo,
            flow_steps=flow_steps,
            sigma_init=sigma_init,
        )
        if transition_noise is None:
            epsilon = torch.randn_like(x_t)
        else:
            epsilon = transition_noise[:, :, index].to(
                device=x_t.device, dtype=torch.float32
            )
        epsilon = epsilon.clamp(-3.0, 3.0)
        x_t = mean + sigma * epsilon
        log_prob = log_prob + _normal_log_prob_sum(x_t, mean, sigma)
        raw_steps.append(x_t)

    raw_chains = torch.stack(raw_steps, dim=2)
    denominator = float((flow_steps + 1) * horizon * action_dim)
    old_chain_score = log_prob / denominator
    raw_final_action, canonical_action = project_ogpo_action_views(
        raw_chains[:, :, -1],
        executed_horizon=executed_horizon,
        active_action_dim=active_action_dim,
    )
    return OGPOChainSample(
        raw_chains=raw_chains,
        old_chain_score=old_chain_score,
        raw_final_action=raw_final_action,
        canonical_action=canonical_action,
    )


def score_ogpo_chains(
    velocity_online_fn: OGPOVelocityFn,
    raw_chains: torch.Tensor,
    *,
    sigma_init: float,
) -> torch.Tensor:
    """Score EMA-generated raw chains under the online action expert.

    Returned shape is ``[B,G]``.  Chain states are detached, while gradients
    through ``velocity_online_fn`` remain available to the online expert.
    """
    if raw_chains.ndim != 5:
        raise ValueError(
            "raw_chains must have shape [B,G,K+1,H,D], got "
            f"{tuple(raw_chains.shape)}"
        )
    _validate_sigma(sigma_init)
    batch_size, group_size, chain_length, horizon, action_dim = raw_chains.shape
    flow_steps = chain_length - 1
    if flow_steps <= 0:
        raise ValueError("raw_chains must contain at least one transition")

    chain = raw_chains.detach().to(dtype=torch.float32)
    log_prob = _normal_log_prob_sum(
        chain[:, :, 0],
        torch.zeros((), device=chain.device, dtype=chain.dtype),
        torch.ones((), device=chain.device, dtype=chain.dtype),
    )
    for index in range(flow_steps):
        x_t = chain[:, :, index]
        x_next = chain[:, :, index + 1]
        time_ogpo = torch.full(
            (batch_size, group_size),
            index / float(flow_steps),
            device=chain.device,
            dtype=torch.float32,
        )
        velocity = velocity_online_fn(x_t, time_ogpo)
        mean, sigma = tapered_ogpo_mean_and_sigma(
            x_t,
            velocity,
            time_ogpo,
            flow_steps=flow_steps,
            sigma_init=sigma_init,
        )
        log_prob = log_prob + _normal_log_prob_sum(x_next, mean, sigma)

    denominator = float((flow_steps + 1) * horizon * action_dim)
    return log_prob / denominator


@torch.no_grad()
def sample_tapered_chains(
    velocity_fn: OGPOVelocityFn,
    initial_noise: torch.Tensor,
    step_noise: torch.Tensor,
    sigma0: float,
    *,
    executed_horizon: int = 10,
    active_action_dim: int = 14,
) -> OGPOChainSample:
    """Public pi0 adapter API for deterministic EMA chain generation.

    ``step_noise`` fixes both the number of flow steps and their innovations,
    which makes sampler/scorer parity directly testable.  All returned chain
    scores are normalized over ``(K+1)*H*D``.
    """
    if step_noise.ndim != 5:
        raise ValueError(
            "step_noise must have shape [B,G,K,H,D], got "
            f"{tuple(step_noise.shape)}"
        )
    return sample_ogpo_chains(
        velocity_fn,
        initial_noise=initial_noise,
        transition_noise=step_noise,
        flow_steps=step_noise.shape[2],
        sigma_init=sigma0,
        executed_horizon=executed_horizon,
        active_action_dim=active_action_dim,
    )


def score_tapered_chains(
    velocity_fn: OGPOVelocityFn,
    chains: torch.Tensor,
    sigma0: float,
) -> torch.Tensor:
    """Public pi0 adapter API for differentiable same-chain scoring."""
    return score_ogpo_chains(
        velocity_fn,
        chains,
        sigma_init=sigma0,
    )


def flow_matching_success_bc_loss(
    predicted_velocity_ogpo: torch.Tensor,
    actions: torch.Tensor,
    noise: torch.Tensor,
    *,
    execution_horizon: int,
) -> torch.Tensor:
    """Pi0 flow-matching BC on the executed model-coordinate prefix."""
    if predicted_velocity_ogpo.shape != actions.shape or actions.shape != noise.shape:
        raise ValueError("predicted velocity, actions, and noise must share [B,H,D]")
    if actions.ndim != 3:
        raise ValueError("success BC tensors must have shape [B,H,D]")
    if not 0 < execution_horizon <= actions.shape[1]:
        raise ValueError("execution_horizon must select a non-empty action prefix")
    # Pi0 uses x_t=t*noise+(1-t)*action and predicts noise-action.  The OGPO
    # velocity adapter reverses that sign, so its matching target is action-noise.
    target_velocity_ogpo = actions - noise
    residual = (
        predicted_velocity_ogpo[:, :execution_horizon]
        - target_velocity_ogpo[:, :execution_horizon]
    )
    return residual.square().mean()


def conservative_ogpo_advantages(q_full: torch.Tensor) -> torch.Tensor:
    """Compute OGPO+CA advantages from ``Q_full[B,G,M]``."""
    if q_full.ndim != 3:
        raise ValueError(
            "q_full must have shape [B,G,M], got " f"{tuple(q_full.shape)}"
        )
    per_head = q_full - q_full.mean(dim=1, keepdim=True)
    lower = per_head.amin(dim=2)
    upper = per_head.amax(dim=2)
    advantages = torch.where(
        lower > 0.0,
        lower,
        torch.where(upper < 0.0, upper, torch.zeros_like(lower)),
    )
    return advantages.detach()


def clipped_ogpo_actor_loss(
    current_chain_score: torch.Tensor,
    old_chain_score: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_epsilon: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Whole-chain PPO surrogate used by OGPO policy extraction."""
    if not 0.0 < clip_epsilon < 1.0:
        raise ValueError(
            f"clip_epsilon must be between zero and one, got {clip_epsilon}"
        )
    if current_chain_score.shape != old_chain_score.shape:
        raise ValueError("current and old chain scores must have the same shape")
    if advantages.shape != current_chain_score.shape:
        raise ValueError("advantages must match the [B,G] chain-score shape")

    old_score = old_chain_score.detach()
    advantage = advantages.detach()
    log_ratio = current_chain_score - old_score
    ratio = torch.exp(log_ratio)
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    loss = -torch.minimum(ratio * advantage, clipped_ratio * advantage).mean()
    with torch.no_grad():
        stats = {
            "ratio": ratio.mean(),
            "ratio_min": ratio.min(),
            "ratio_max": ratio.max(),
            "approx_kl": ((ratio - 1.0) - log_ratio).mean(),
        }
    return loss, stats
