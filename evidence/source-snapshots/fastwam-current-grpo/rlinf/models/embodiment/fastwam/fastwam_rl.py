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

"""Batch-native Fast-WAM ODE/Flow-SDE core for RLinf.

The upstream Fast-WAM scheduler is authoritative for the shifted inference grid and
the model timestep.  RLinf's OpenPI implementation is authoritative for the one-step
Flow-SDE transition.  Evaluation, rollout and actor replay share the same
conditioning, velocity and schedule primitives; only evaluation omits the stochastic
transition and replay tensors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ActionSchedule:
    """Resolved official action schedule.

    ``timesteps`` and ``deltas`` retain the model dtype used by the official
    scheduler.  Normalized time quantities are FP32 for the probability path.
    """

    timesteps: torch.Tensor
    deltas: torch.Tensor
    normalized_timesteps: torch.Tensor
    next_normalized_timesteps: torch.Tensor
    dts: torch.Tensor
    configured_shift: float | None
    effective_shift: float


@dataclass(frozen=True)
class RolloutRandomness:
    """Random tensors generated once for a complete logical rollout batch."""

    initial_latents: torch.Tensor
    denoise_inds: torch.Tensor
    sde_epsilon: torch.Tensor

    def select(self, item: slice) -> "RolloutRandomness":
        return RolloutRandomness(
            initial_latents=self.initial_latents[item],
            denoise_inds=self.denoise_inds[item],
            sde_epsilon=self.sde_epsilon[item],
        )


@dataclass(frozen=True)
class ActionConditioning:
    """Frozen first-frame conditioning reused across action denoise steps."""

    context: torch.Tensor
    context_mask: torch.Tensor
    video_cache_k: list[torch.Tensor]
    video_cache_v: list[torch.Tensor]
    action_attention_mask: torch.Tensor


@dataclass(frozen=True)
class FlowRollout:
    """Output of the shared ODE/Flow-SDE loop."""

    actions: torch.Tensor
    chains: torch.Tensor | None = None
    prev_logprobs: torch.Tensor | None = None
    denoise_inds: torch.Tensor | None = None


def _model_device(model) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device


def _model_dtype(model) -> torch.dtype:
    dtype = getattr(model, "torch_dtype", None)
    if dtype is not None:
        return dtype
    return next(model.parameters()).dtype


def _expand_batch_vector(value: torch.Tensor, batch_size: int) -> torch.Tensor:
    if value.ndim == 0:
        return value.reshape(1).expand(batch_size)
    if value.ndim != 1 or value.shape[0] != batch_size:
        raise ValueError(
            f"Expected scalar or [B={batch_size}] tensor, got {tuple(value.shape)}"
        )
    return value


def _right_broadcast(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    while value.ndim < reference.ndim:
        value = value.unsqueeze(-1)
    return value


def resolve_action_schedule(
    model,
    num_inference_steps: int,
    sigma_shift: float | None,
) -> ActionSchedule:
    """Resolve the official shifted schedule and validate its denoise direction."""

    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    scheduler = model.infer_action_scheduler
    timesteps, deltas = scheduler.build_inference_schedule(
        num_inference_steps=num_inference_steps,
        device=_model_device(model),
        dtype=_model_dtype(model),
        shift_override=sigma_shift,
    )
    if timesteps.shape != (num_inference_steps,) or deltas.shape != (
        num_inference_steps,
    ):
        raise ValueError(
            "Official action schedule returned unexpected shapes: "
            f"timesteps={tuple(timesteps.shape)}, deltas={tuple(deltas.shape)}"
        )

    deltas_fp32 = deltas.float()
    if not bool(torch.isfinite(timesteps.float()).all()) or not bool(
        torch.isfinite(deltas_fp32).all()
    ):
        raise FloatingPointError("Action schedule contains NaN/Inf")
    if not bool((deltas_fp32 < 0).all()):
        raise ValueError("Fast-WAM denoise schedule must be strictly decreasing")

    num_train_timesteps = float(scheduler.num_train_timesteps)
    normalized = timesteps.float() / num_train_timesteps
    next_normalized = normalized + deltas_fp32
    if not bool((normalized > 0).all()) or not bool((normalized <= 1).all()):
        raise ValueError("Normalized Fast-WAM timesteps must lie in (0, 1]")
    if not bool((next_normalized >= -1e-6).all()) or not bool(
        (next_normalized < normalized).all()
    ):
        raise ValueError("Invalid next points in Fast-WAM action schedule")

    effective_shift = (
        float(scheduler.shift) if sigma_shift is None else float(sigma_shift)
    )
    return ActionSchedule(
        timesteps=timesteps,
        deltas=deltas,
        normalized_timesteps=normalized,
        next_normalized_timesteps=next_normalized.clamp_min(0.0),
        dts=-deltas_fp32,
        configured_shift=None if sigma_shift is None else float(sigma_shift),
        effective_shift=effective_shift,
    )


def prepare_initial_action_latents(
    *,
    batch_size: int,
    action_horizon: int,
    action_dim: int,
    device: torch.device | str,
    dtype: torch.dtype,
    rand_device: torch.device | str = "cpu",
    seed: int | None = None,
    broadcast_singleton: bool = False,
) -> torch.Tensor:
    """Create official-style action noise.

    With ``broadcast_singleton=True`` this is the vectorized equivalent of calling
    upstream B=1 ``infer_action(seed=seed)`` independently for every environment.
    Training never uses this option.
    """

    if batch_size <= 0 or action_horizon <= 0 or action_dim <= 0:
        raise ValueError("batch_size, action_horizon and action_dim must be positive")
    if broadcast_singleton and seed is None:
        raise ValueError("A fixed seed is required for singleton-broadcast evaluation")

    rand_device = torch.device(rand_device)
    generator = (
        None
        if seed is None
        else torch.Generator(device=rand_device).manual_seed(int(seed))
    )
    draw_batch = 1 if broadcast_singleton else batch_size
    latents = torch.randn(
        (draw_batch, action_horizon, action_dim),
        generator=generator,
        device=rand_device,
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)
    if broadcast_singleton:
        latents = latents.expand(batch_size, -1, -1).clone()
    return latents


def prepare_rollout_randomness(
    *,
    batch_size: int,
    action_horizon: int,
    action_dim: int,
    num_inference_steps: int,
    device: torch.device | str,
    rand_device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> RolloutRandomness:
    """Generate independent trajectory noise and one OpenPI-style shared index."""

    if batch_size <= 0 or action_horizon <= 0 or action_dim <= 0:
        raise ValueError("batch_size, action_horizon and action_dim must be positive")
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    rand_device = torch.device(rand_device)
    if generator is not None and torch.device(generator.device) != rand_device:
        raise ValueError("generator device must match rand_device")
    initial_latents = torch.randn(
        (batch_size, action_horizon, action_dim),
        generator=generator,
        device=rand_device,
        dtype=torch.float32,
    )
    sde_epsilon = torch.randn(
        (batch_size, action_horizon, action_dim),
        generator=generator,
        device=rand_device,
        dtype=torch.float32,
    )
    shared_k = torch.randint(
        0,
        num_inference_steps,
        (1,),
        generator=generator,
        device=rand_device,
        dtype=torch.long,
    )
    denoise_inds = shared_k.expand(batch_size).clone()
    return RolloutRandomness(
        initial_latents=initial_latents.to(device=device),
        denoise_inds=denoise_inds.to(device=device),
        sde_epsilon=sde_epsilon.to(device=device),
    )


@torch.no_grad()
def encode_first_frame_latents(
    model,
    input_image: torch.Tensor,
    *,
    tiled: bool = False,
) -> torch.Tensor:
    """Encode a real batch of one-frame videos with the current tensor VAE."""

    if input_image.ndim != 4 or input_image.shape[1] != 3:
        raise ValueError(
            f"input_image must be [B,3,H,W], got {tuple(input_image.shape)}"
        )
    if tiled:
        raise NotImplementedError(
            "Current Fast-WAM batched first-frame encoding does not support tiling"
        )
    video = input_image.to(device=_model_device(model)).unsqueeze(2)
    latents = model.vae.model.encode(video, model.vae.scale)
    if not torch.is_tensor(latents) or latents.shape[0] != input_image.shape[0]:
        raise ValueError("Official Fast-WAM VAE wrapper returned an invalid batch")
    return latents


@torch.no_grad()
def build_action_conditioning(
    model,
    *,
    input_image: torch.Tensor,
    text_context: torch.Tensor,
    text_context_mask: torch.Tensor,
    proprio: torch.Tensor,
    action_horizon: int,
    tiled: bool = False,
) -> ActionConditioning:
    """Mirror current upstream action inference through its tensor KV cache."""

    batch_size = input_image.shape[0]
    if text_context.ndim != 3 or text_context.shape[0] != batch_size:
        raise ValueError("text_context must be [B,L,D] and match image batch")
    if text_context_mask.shape != text_context.shape[:2]:
        raise ValueError("text_context_mask must match text_context [B,L]")
    if proprio.ndim != 2 or proprio.shape[0] != batch_size:
        raise ValueError("proprio must be [B,D] and match image batch")

    device = _model_device(model)
    model_dtype = _model_dtype(model)
    base_context = text_context.to(device=device, dtype=model_dtype)
    base_mask = text_context_mask.to(device=device, dtype=torch.bool)
    context, context_mask = model._append_proprio_to_context(
        context=base_context,
        context_mask=base_mask,
        proprio=proprio.to(device=device, dtype=torch.float32),
    )

    first_frame_latents = encode_first_frame_latents(
        model,
        input_image.to(device=device, dtype=model_dtype),
        tiled=tiled,
    )
    timestep_video = torch.zeros(
        (batch_size,),
        dtype=first_frame_latents.dtype,
        device=device,
    )
    (
        video_tokens,
        _t_video,
        video_t_mod,
        video_context,
        video_context_mask,
        video_freqs,
        _f_video,
        _h_video,
        _w_video,
        tokens_per_frame,
    ) = model.video_expert.prepare(
        x=first_frame_latents,
        timestep=timestep_video,
        context=context,
        context_mask=context_mask,
        action=None,
        fuse_vae_embedding_in_latents=bool(
            getattr(model.video_expert, "fuse_vae_embedding_in_latents", False)
        ),
    )
    video_seq_len = int(video_tokens.shape[1])
    attention_mask = model._build_mot_attention_mask(
        video_seq_len=video_seq_len,
        action_seq_len=action_horizon,
        video_tokens_per_frame=int(tokens_per_frame),
        device=video_tokens.device,
    )
    video_cache_k, video_cache_v = model.mot.prefill_video_cache_tensor(
        video_tokens=video_tokens,
        video_freqs=video_freqs,
        video_t_mod=video_t_mod,
        video_context=video_context,
        video_context_mask=video_context_mask,
        video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
    )
    return ActionConditioning(
        context=context,
        context_mask=context_mask,
        video_cache_k=video_cache_k,
        video_cache_v=video_cache_v,
        action_attention_mask=attention_mask[video_seq_len:, :],
    )


def predict_action_velocity(
    model,
    *,
    x: torch.Tensor,
    raw_timestep: torch.Tensor,
    conditioning: ActionConditioning,
) -> torch.Tensor:
    """Grad-enabled twin of upstream ``_predict_action_noise_with_cache``."""

    batch_size = x.shape[0]
    device = _model_device(model)
    model_dtype = _model_dtype(model)
    timestep = _expand_batch_vector(raw_timestep, batch_size).to(
        device=device, dtype=model_dtype
    )
    (
        action_tokens,
        _t_action,
        action_t_mod,
        action_context,
        action_context_mask,
        action_freqs,
    ) = model.action_expert.prepare(
        action_tokens=x.to(device=device, dtype=model_dtype),
        timestep=timestep,
        context=conditioning.context,
        context_mask=conditioning.context_mask,
    )
    action_tokens = model.mot.forward_action_with_video_cache_tensor(
        action_tokens=action_tokens,
        action_freqs=action_freqs,
        action_t_mod=action_t_mod,
        action_context=action_context,
        action_context_mask=action_context_mask,
        video_cache_k=conditioning.video_cache_k,
        video_cache_v=conditioning.video_cache_v,
        action_attention_mask=conditioning.action_attention_mask,
    )
    return model.action_expert.post(action_tokens).float()


def flow_step_mean_std(
    *,
    x: torch.Tensor,
    velocity: torch.Tensor,
    normalized_timestep: torch.Tensor,
    next_normalized_timestep: torch.Tensor,
    signed_delta: torch.Tensor,
    noise_level: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """OpenPI Flow-SDE mean/std on the official Fast-WAM shifted grid."""

    if noise_level <= 0:
        raise ValueError("noise_level must be positive for a stochastic transition")
    x = x.float()
    velocity = velocity.float()
    batch_size = x.shape[0]
    t = _right_broadcast(
        _expand_batch_vector(normalized_timestep.float(), batch_size).to(x.device),
        x,
    )
    next_t = _right_broadcast(
        _expand_batch_vector(next_normalized_timestep.float(), batch_size).to(x.device),
        x,
    )
    delta = _right_broadcast(
        _expand_batch_vector(signed_delta.float(), batch_size).to(x.device), x
    )
    if not bool((delta < 0).all()):
        raise ValueError("signed_delta must be strictly negative")
    dt = -delta
    if not bool((t > 0).all()) or not bool((t <= 1).all()):
        raise ValueError("normalized_timestep must lie in (0, 1]")

    first_step = torch.isclose(t, torch.ones_like(t), rtol=0.0, atol=1e-6)
    denominator_t = torch.where(first_step, next_t, t)
    one_minus_denominator = 1.0 - denominator_t
    if not bool((one_minus_denominator > 0).all()):
        raise ValueError("Flow-SDE sigma denominator must be strictly positive")

    x0 = x - t * velocity
    x1 = x + (1.0 - t) * velocity
    sigma = float(noise_level) * torch.sqrt(t / one_minus_denominator)
    mean = x0 * (1.0 - (t - dt)) + x1 * (t - dt - sigma.square() * dt / (2.0 * t))
    std = (torch.sqrt(dt) * sigma).expand_as(mean)
    mean_ode = x + delta * velocity
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
        raise FloatingPointError("Flow-SDE mean/std contains NaN/Inf")
    if not bool((std > 0).all()):
        raise ValueError("Flow-SDE std must be strictly positive")
    return mean, std, mean_ode


def gaussian_logprob(
    sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Elementwise FP32 Gaussian log density without a density-changing floor."""

    sample, mean, std = sample.float(), mean.float(), std.float()
    if not bool(torch.isfinite(std).all()) or not bool((std > 0).all()):
        raise ValueError("Gaussian std must be finite and strictly positive")
    return (
        -0.5 * ((sample - mean) / std).square()
        - torch.log(std)
        - 0.5 * math.log(2.0 * math.pi)
    )


def gaussian_entropy(std: torch.Tensor) -> torch.Tensor:
    """Elementwise FP32 Gaussian differential entropy."""

    std = std.float()
    if not bool(torch.isfinite(std).all()) or not bool((std > 0).all()):
        raise ValueError("Gaussian std must be finite and strictly positive")
    return torch.log(std) + 0.5 * math.log(2.0 * math.pi * math.e)


def flow_sde_rollout(
    model,
    *,
    conditioning: ActionConditioning,
    initial_latents: torch.Tensor,
    num_inference_steps: int,
    sigma_shift: float | None,
    noise_level: float,
    deterministic: bool,
    denoise_inds: torch.Tensor | None = None,
    sde_epsilon: torch.Tensor | None = None,
) -> FlowRollout:
    """Run the one shared denoise loop.

    This function deliberately contains no RNG.  The policy creates all random
    tensors once for the complete logical batch before resource chunking.
    """

    schedule = resolve_action_schedule(model, num_inference_steps, sigma_shift)
    batch_size = initial_latents.shape[0]
    if initial_latents.ndim != 3:
        raise ValueError("initial_latents must be [B,H,D]")

    if deterministic:
        x = initial_latents.to(device=_model_device(model), dtype=_model_dtype(model))
        chains = None
    else:
        if denoise_inds is None or sde_epsilon is None:
            raise ValueError("Stochastic rollout requires denoise_inds and sde_epsilon")
        if denoise_inds.shape != (batch_size,):
            raise ValueError("denoise_inds must be [B]")
        if sde_epsilon.shape != initial_latents.shape:
            raise ValueError("sde_epsilon must match initial_latents")
        denoise_inds = denoise_inds.to(device=_model_device(model), dtype=torch.long)
        if not bool(((denoise_inds >= 0) & (denoise_inds < num_inference_steps)).all()):
            raise ValueError("denoise_inds contains an out-of-range step")
        x = initial_latents.to(device=_model_device(model), dtype=torch.float32)
        sde_epsilon = sde_epsilon.to(device=x.device, dtype=torch.float32)
        chains = [x]
        selected_logprob = torch.zeros_like(x)
        selected_seen = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

    for step_index in range(num_inference_steps):
        velocity = predict_action_velocity(
            model,
            x=x,
            raw_timestep=schedule.timesteps[step_index],
            conditioning=conditioning,
        )
        if deterministic:
            x = model.infer_action_scheduler.step(
                velocity.to(dtype=x.dtype),
                schedule.deltas[step_index],
                x,
            )
            continue

        mean, std, mean_ode = flow_step_mean_std(
            x=x,
            velocity=velocity,
            normalized_timestep=schedule.normalized_timesteps[step_index],
            next_normalized_timestep=schedule.next_normalized_timesteps[step_index],
            signed_delta=schedule.deltas[step_index],
            noise_level=noise_level,
        )
        selected = denoise_inds == step_index
        selected_view = selected.view(batch_size, 1, 1)
        sampled = mean + std * sde_epsilon
        sampled_logprob = gaussian_logprob(sampled, mean, std)
        x = torch.where(selected_view, sampled, mean_ode)
        selected_logprob = torch.where(selected_view, sampled_logprob, selected_logprob)
        selected_seen |= selected
        chains.append(x)

    if deterministic:
        return FlowRollout(actions=x.float())
    if not bool(selected_seen.all()):
        raise RuntimeError("Every trajectory must execute exactly one stochastic step")
    return FlowRollout(
        actions=x.float(),
        chains=torch.stack(chains, dim=1).float(),
        prev_logprobs=selected_logprob.float(),
        denoise_inds=denoise_inds,
    )


def recompute_logprob(
    model,
    *,
    input_image: torch.Tensor,
    text_context: torch.Tensor,
    text_context_mask: torch.Tensor,
    proprio: torch.Tensor,
    chains: torch.Tensor,
    denoise_inds: torch.Tensor,
    action_horizon: int,
    num_inference_steps: int,
    sigma_shift: float | None,
    noise_level: float,
    tiled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rebuild frozen conditioning and replay the real behavior transition."""

    if chains.ndim != 4:
        raise ValueError("chains must be [B,S+1,H,D]")
    batch_size = chains.shape[0]
    if chains.shape[1] != num_inference_steps + 1:
        raise ValueError("chains has the wrong number of denoise states")
    if chains.shape[2] != action_horizon:
        raise ValueError("chains has the wrong action horizon")
    if denoise_inds.shape != (batch_size,):
        raise ValueError("denoise_inds must be [B]")

    conditioning = build_action_conditioning(
        model,
        input_image=input_image,
        text_context=text_context,
        text_context_mask=text_context_mask,
        proprio=proprio,
        action_horizon=action_horizon,
        tiled=tiled,
    )
    schedule = resolve_action_schedule(model, num_inference_steps, sigma_shift)
    denoise_inds = denoise_inds.to(device=chains.device, dtype=torch.long)
    if not bool(((denoise_inds >= 0) & (denoise_inds < num_inference_steps)).all()):
        raise ValueError("denoise_inds contains an out-of-range step")
    rows = torch.arange(batch_size, device=chains.device)
    x = chains[rows, denoise_inds].float()
    realized_next = chains[rows, denoise_inds + 1].float()
    velocity = predict_action_velocity(
        model,
        x=x,
        raw_timestep=schedule.timesteps[denoise_inds],
        conditioning=conditioning,
    )
    mean, std, _ = flow_step_mean_std(
        x=x,
        velocity=velocity,
        normalized_timestep=schedule.normalized_timesteps[denoise_inds],
        next_normalized_timestep=schedule.next_normalized_timesteps[denoise_inds],
        signed_delta=schedule.deltas[denoise_inds],
        noise_level=noise_level,
    )
    logprob = gaussian_logprob(realized_next, mean, std)
    entropy = gaussian_entropy(std)
    return logprob, entropy


__all__ = [
    "ActionConditioning",
    "ActionSchedule",
    "FlowRollout",
    "RolloutRandomness",
    "build_action_conditioning",
    "encode_first_frame_latents",
    "flow_sde_rollout",
    "flow_step_mean_std",
    "gaussian_entropy",
    "gaussian_logprob",
    "predict_action_velocity",
    "prepare_initial_action_latents",
    "prepare_rollout_randomness",
    "recompute_logprob",
    "resolve_action_schedule",
]
