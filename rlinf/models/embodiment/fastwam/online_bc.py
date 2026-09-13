# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Native action-only Fast-WAM FM for success-filtered online BC.

Each replay query supervises the 24 commands submitted to RoboTwin. The model
still predicts 32 positions; the unsubmitted tail is zero padded in normalized
coordinates and excluded from the loss. No video or KV cache is stored in replay.
"""

from __future__ import annotations

import torch

from .fastwam_rl import (
    build_action_conditioning,
    flow_sde_rollout,
    predict_action_velocity,
    prepare_initial_action_latents,
)
from .robotwin_adapter import adapt_robotwin_observation, denormalize_actions


FASTWAM_BC_OBSERVATION_KEYS = (
    "image", "text_context", "text_context_mask", "proprio"
)


def prepare_command_targets(commands, valid_mask, processor, *, horizon=32):
    """Normalize submitted commands with release stats and mask the model tail."""
    commands = torch.as_tensor(commands).detach().to("cpu", torch.float32)
    if commands.ndim != 3 or commands.shape[1:] != (24, 14):
        raise ValueError("Fast-WAM BC labels must be submitted [B,24,14] commands")
    mask = torch.as_tensor(valid_mask, dtype=torch.bool, device="cpu")
    if mask.shape != commands.shape or not mask.flatten(1).any(1).all():
        raise ValueError("Fast-WAM command mask must match labels and be nonempty")
    if horizon != 32 or not torch.isfinite(commands).all():
        raise ValueError("Fast-WAM BC requires H32 and finite submitted commands")
    # The pinned RoboTwin processor has no relative-action transform. This is
    # the inverse of robotwin_adapter.denormalize_actions, in the same stats.
    normalizer = processor.normalizer.normalizers["action"]["default"]
    normalized = normalizer.forward(commands)
    if normalized.shape != commands.shape or not torch.isfinite(normalized).all():
        raise ValueError("Official action normalization returned invalid targets")
    targets = torch.zeros(commands.shape[0], horizon, 14, dtype=torch.float32)
    targets[:, :24] = normalized
    padded_mask = torch.zeros_like(targets, dtype=torch.bool)
    padded_mask[:, :24] = mask
    return targets, padded_mask


def native_action_fm_loss(prediction, target, valid_mask, timestep, scheduler):
    """Official per-query valid-action reduction and timestep weighting."""
    if prediction.shape != target.shape or valid_mask.shape != target.shape:
        raise ValueError("FM prediction, target and command mask must match")
    mask = valid_mask.to(device=prediction.device, dtype=torch.float32)
    denominator = mask.sum(dim=(1, 2))
    if (denominator <= 0).any():
        raise ValueError("Every BC query needs a valid submitted target")
    mse = (prediction.float() - target.float()).square()
    per_query = (mse * mask).sum(dim=(1, 2)) / denominator
    weights = scheduler.training_weight(timestep).to(per_query)
    return (per_query * weights).mean()


class FastWAMOnlineBCMixin:
    """Optional BC path; the existing GRPO inference/replay path is unchanged."""

    @torch.no_grad()
    def predict_online_bc_action_batch(self, env_obs, mode="train"):
        if mode not in ("train", "eval"):
            raise ValueError(f"Invalid online BC rollout mode: {mode}")
        image, proprio, prompts = adapt_robotwin_observation(
            env_obs, self.processor, device=self.device, dtype=self.model_dtype
        )
        context, context_mask = self.model.encode_prompt(prompts)
        batch_size = image.shape[0]
        # HuggingFaceWorker seeds the continuous training stream once and saves /
        # restores it around evaluation. Do not reseed every query or broadcast
        # identical noise across the batch, as native fixed-seed eval would do.
        initial = prepare_initial_action_latents(
            batch_size=batch_size,
            action_horizon=self.config.action_horizon,
            action_dim=self.config.action_dim,
            device=self.device,
            dtype=self.model_dtype,
            rand_device=self.config.rand_device,
        )
        normalized = []
        micro = self.config.model_forward_batch_size
        for start in range(0, batch_size, micro):
            item = slice(start, min(start + micro, batch_size))
            conditioning = build_action_conditioning(
                self.model,
                input_image=image[item],
                text_context=context[item],
                text_context_mask=context_mask[item],
                proprio=proprio[item],
                action_horizon=self.config.action_horizon,
                tiled=self.config.tiled,
            )
            result = flow_sde_rollout(
                self.model,
                conditioning=conditioning,
                initial_latents=initial[item],
                num_inference_steps=self.config.num_inference_steps,
                sigma_shift=self.config.sigma_shift,
                noise_level=self.config.noise_level,
                deterministic=True,
            )
            normalized.append(result.actions.detach().float().cpu())
        actions = denormalize_actions(torch.cat(normalized), self.processor)[:, :24]
        inputs = {
            "image": image.detach().cpu().contiguous(),
            "text_context": context.detach().cpu().contiguous(),
            "text_context_mask": context_mask.detach().bool().cpu().contiguous(),
            "proprio": proprio.detach().float().cpu().contiguous(),
        }
        return actions, {
            "prev_logprobs": torch.zeros_like(actions),
            "prev_values": None,
            "forward_inputs": inputs,
        }

    def prepare_dagger_sft_batch(self, batch):
        batch = batch.get("forward_inputs", batch)
        missing = set(FASTWAM_BC_OBSERVATION_KEYS).difference(batch)
        if missing:
            raise ValueError(f"Fast-WAM BC replay observations missing: {sorted(missing)}")
        commands = batch["action"].reshape(-1, 24, 14)
        targets, valid = prepare_command_targets(
            commands, batch["action_valid_mask"], self.processor,
            horizon=self.config.action_horizon,
        )
        return {
            **{key: batch[key] for key in FASTWAM_BC_OBSERVATION_KEYS},
            "targets": targets,
            "valid_mask": valid,
        }

    def sft_forward(self, data, *, noise=None, timestep=None, **kwargs):
        """Use release train scheduler and action velocity; freeze conditioners.

        Explicit noise/timestep are only for the fixed-input oracle check. Normal
        training samples both from the actor RNG, exactly as upstream FM does.
        """
        actions = data["targets"].to(device=self.device, dtype=self.model_dtype)
        valid = data["valid_mask"].to(self.device)
        scheduler = self.model.train_action_scheduler
        if noise is None:
            noise = torch.randn_like(actions)
        else:
            noise = noise.to(actions)
        if timestep is None:
            timestep = scheduler.sample_training_t(
                batch_size=actions.shape[0], device=actions.device, dtype=actions.dtype
            )
        else:
            timestep = timestep.to(actions)
        noisy = scheduler.add_noise(actions, noise, timestep)
        target = scheduler.training_target(actions, noise, timestep)
        conditioning = build_action_conditioning(
            self.model,
            input_image=data["image"],
            text_context=data["text_context"],
            text_context_mask=data["text_context_mask"],
            proprio=data["proprio"],
            action_horizon=self.config.action_horizon,
            tiled=self.config.tiled,
        )
        prediction = predict_action_velocity(
            self.model, x=noisy, raw_timestep=timestep, conditioning=conditioning
        )
        return float(self.model.loss_lambda_action) * native_action_fm_loss(
            prediction, target, valid, timestep, scheduler
        )
