"""Batch-only inference adapter for OpenDW e33befa's released RobotWin model.

The upstream MoT/action experts and Wan VAE already retain a batch dimension;
the public inference wrapper restricts images, proprio and history to B1.  This
adapter lifts those entry checks without changing normalization, denoising or
history alignment.  Every diffusion step is one shared-model call with B rows.
The original ``infer_joint`` remains available for B1 GPU comparison.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from types import MethodType
from typing import Any

import torch
from PIL import Image


MAX_WM_BATCH = 16


def _image(model, value: torch.Tensor, *, batch_size: int | None = None) -> torch.Tensor:
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if value.ndim != 4 or value.shape[1] != 3:
        raise ValueError(f"Images must be [B,3,H,W], got {tuple(value.shape)}")
    if batch_size is not None and value.shape[0] != batch_size:
        raise ValueError("Every history/current frame must have the same batch")
    return value.to(device=model.device, dtype=model.torch_dtype)


def _noise(model, shape: tuple[int, ...], seeds: Sequence[int], rand_device: str) -> torch.Tensor:
    # Upstream creates separate generators with the same row seed for action
    # and video. Drawing [1,...] independently preserves that B1 random stream,
    # including when the row moves to another microbatch or a short tail.
    return torch.cat([
        torch.randn(
            (1, *shape),
            generator=torch.Generator(device=rand_device).manual_seed(int(seed)),
            device=rand_device,
            dtype=torch.float32,
        )
        for seed in seeds
    ], dim=0).to(device=model.device, dtype=model.torch_dtype)


def _prepare_context(
    model, *, batch_size: int, prompt, context, context_mask, proprio,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prompt is not None:
        if context is not None or context_mask is not None:
            raise ValueError("prompt and context/context_mask are mutually exclusive")
        if isinstance(prompt, str):
            context, context_mask = model.encode_prompt(prompt)
        else:
            if len(prompt) != batch_size:
                raise ValueError("One prompt per batch row is required")
            context, context_mask = model.encode_prompt(list(prompt))
    elif context is None or context_mask is None:
        raise ValueError("Provide prompt or both context and context_mask")
    if context.ndim == 2:
        context = context.unsqueeze(0)
    if context_mask.ndim == 1:
        context_mask = context_mask.unsqueeze(0)
    if context.ndim != 3 or context_mask.ndim != 2:
        raise ValueError("Context/mask must be [B,L,D]/[B,L]")
    if tuple(context.shape[:2]) != tuple(context_mask.shape):
        raise ValueError("Context and context_mask shapes do not match")
    if context.shape[0] == 1 and batch_size > 1:
        context = context.expand(batch_size, -1, -1)
        context_mask = context_mask.expand(batch_size, -1)
    elif context.shape[0] != batch_size:
        raise ValueError("Context batch must be one or equal the image batch")
    context = context.to(device=model.device, dtype=model.torch_dtype)
    context_mask = context_mask.to(device=model.device, dtype=torch.bool)
    if proprio is not None:
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        if model.proprio_dim is None:
            raise ValueError("proprio supplied to model without a proprio encoder")
        if tuple(proprio.shape) != (batch_size, model.proprio_dim):
            raise ValueError(f"proprio must be [{batch_size},{model.proprio_dim}]")
        context, context_mask = model._append_proprio_to_context(
            context=context,
            context_mask=context_mask,
            proprio=proprio.to(device=model.device, dtype=model.torch_dtype),
        )
    return context, context_mask


def _encode_conditions(model, input_image, input_images, batch_size: int, clean_count: int):
    vae = model.vae
    if clean_count <= 1:
        # Match the H1 upstream fallback: use input_image when provided.
        return vae.single_encode(input_image.unsqueeze(2), model.device)
    history_frames = (clean_count - 1) * int(vae.temporal_downsample_factor)
    if input_images is None:
        images = [input_image] * (history_frames + 1)
    else:
        images = list(input_images)
        if len(images) == history_frames:
            images.append(input_image)
        if len(images) != history_frames + 1:
            raise ValueError(f"Expected {history_frames} history frames plus current frame")
        images = [_image(model, frame, batch_size=batch_size) for frame in images]
        if any(frame.shape != input_image.shape for frame in images):
            raise ValueError("History/current image spatial shapes must match")
    condition_video = torch.stack(images, dim=2)
    history_video = condition_video[:, :, :history_frames]
    # Same causal anchor and discarded first latent as DW05History.
    history_for_vae = torch.cat([history_video[:, :, :1], history_video], dim=2)
    history_latents = vae.single_encode(history_for_vae, model.device)[:, :, 1:]
    current_latents = vae.single_encode(condition_video[:, :, history_frames:], model.device)
    conditions = torch.cat([history_latents, current_latents], dim=2)
    if conditions.shape[0] != batch_size or conditions.shape[2] != clean_count:
        raise RuntimeError("VAE returned an unexpected history/current latent shape")
    return conditions


@torch.no_grad()
def infer_joint_batch(
    model,
    *,
    input_image: torch.Tensor,
    seeds: Sequence[int],
    action: torch.Tensor,
    proprio: torch.Tensor | None = None,
    context: torch.Tensor | None = None,
    context_mask: torch.Tensor | None = None,
    prompt: str | Sequence[str] | None = None,
    input_images: Sequence[torch.Tensor] | None = None,
    num_video_frames: int = 9,
    action_horizon: int = 32,
    num_inference_steps: int = 10,
    sigma_shift: float | None = None,
    rand_device: str = "cpu",
    tiled: bool = False,
) -> dict[str, Any]:
    """Return ``video[row][frame]`` PIL images and CPU ``action[B,C,D]``.

    Images/actions/proprio must already have the exact upstream preprocessing.
    Batch sizes 1..16, including final partial groups, use the same tensor path.
    No automatic OOM downshift or serial denoising fallback is performed.
    """
    if tiled:
        raise ValueError("The validated RobotWin contract uses tiled=False")
    if (num_video_frames, action_horizon, num_inference_steps) != (9, 32, 10):
        raise ValueError("This adapter preserves the deployed video9/C32/10-step contract")
    input_image = _image(model, input_image)
    batch_size, _, height, width = input_image.shape
    if not 1 <= batch_size <= MAX_WM_BATCH:
        raise ValueError(f"Batch must be 1..{MAX_WM_BATCH}, got {batch_size}")
    if len(seeds) != batch_size or any(
        isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0 or seed >= 2**64
        for seed in seeds
    ):
        raise ValueError("Each batch row needs one independent nonnegative uint64 seed")
    if model._check_resize_height_width(height, width, num_video_frames) != (height, width, num_video_frames):
        raise ValueError("Images must already be aligned to the upstream VAE grid")
    action = model._normalize_infer_action_condition(
        action, batch_size=batch_size, action_horizon=action_horizon, name="action",
    )
    if action is None:
        raise ValueError("Externally supplied actions are required for WM inference")
    clean_count = int(getattr(model, "num_hist_frames", 1))
    if clean_count < 1:
        raise ValueError("num_hist_frames must be positive")
    model.eval()
    try:
        conditions = _encode_conditions(model, input_image, input_images, batch_size, clean_count)
        temporal = int(model.vae.temporal_downsample_factor)
        spatial = int(model.vae.upsampling_factor)
        main_latent_t = (num_video_frames - 1) // temporal + 1
        latent_shape = (model.vae.model.z_dim, clean_count - 1 + main_latent_t,
                        height // spatial, width // spatial)
        latents_video = _noise(model, latent_shape, seeds, rand_device)
        latents_action = _noise(model, (action_horizon, model.action_expert.action_dim), seeds, rand_device)
        latents_video[:, :, :clean_count] = conditions
        context, context_mask = _prepare_context(
            model, batch_size=batch_size, prompt=prompt,
            context=context, context_mask=context_mask, proprio=proprio,
        )
        timesteps_video, deltas_video = model.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps, device=model.device,
            dtype=latents_video.dtype, shift_override=sigma_shift,
        )
        timesteps_action, deltas_action = model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps, device=model.device,
            dtype=latents_action.dtype, shift_override=sigma_shift,
        )
        if len(timesteps_video) != num_inference_steps or len(timesteps_action) != num_inference_steps:
            raise RuntimeError("Inference schedule length changed")
        fuse_flag = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))
        denoiser_batch_sizes = []
        for tv, dv, ta, da in zip(timesteps_video, deltas_video, timesteps_action, deltas_action):
            # Call the batch-capable WorldActionModel method directly. Its
            # History shim only adds a B1 input check, which is handled above.
            pred_video, pred_action = model._predict_joint_noise(
                latents_video=latents_video, latents_action=latents_action,
                timestep_video=tv.reshape(1).expand(batch_size).to(latents_video),
                timestep_action=ta.reshape(1).expand(batch_size).to(latents_action),
                context=context, context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                clean_latent_count=clean_count, action_condition_start_latent=clean_count,
                gt_action=action,
            )
            if pred_video.shape != latents_video.shape or pred_action.shape != latents_action.shape:
                raise RuntimeError("Denoiser changed the video/action batch shape")
            denoiser_batch_sizes.append(int(pred_video.shape[0]))
            latents_video = model.infer_video_scheduler.step(pred_video, dv, latents_video)
            latents_action = model.infer_action_scheduler.step(pred_action, da, latents_action)
            latents_video[:, :, :clean_count] = conditions
        main_latents = latents_video[:, :, clean_count - 1:]
        # single_decode means one VAE call, not one sample: it accepts the full
        # B dimension, whereas public vae.decode loops over B and copies to CPU.
        decoded = model.vae.single_decode(main_latents, model.device)
        if tuple(decoded.shape) != (batch_size, 3, num_video_frames, height, width):
            raise RuntimeError(f"Unexpected decoded video shape: {tuple(decoded.shape)}")
        if not bool(torch.isfinite(decoded).all()) or not bool(torch.isfinite(latents_action).all()):
            raise RuntimeError("Non-finite WM video/action output")
        pixels = ((decoded.detach().float().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8).cpu()
        videos = [[Image.fromarray(pixels[row, :, frame].permute(1, 2, 0).numpy())
                   for frame in range(num_video_frames)] for row in range(batch_size)]
        return {
            "video": videos,
            "action": latents_action.detach().to(device="cpu", dtype=torch.float32),
            "batch_size": batch_size,
            "denoiser_batch_sizes": denoiser_batch_sizes,
            "decoded_video_shape": list(decoded.shape),
            "action_shape": list(latents_action.shape),
        }
    finally:
        # Upstream normally clears caches at entry/exit; an OOM inside VAE
        # must also drop partial feature maps before the owner unloads it.
        model.vae.model.clear_cache()


def install_batch_inference(model) -> None:
    """Expose both paths explicitly; never replace the public B1 entrypoint."""
    if not hasattr(model, "infer_joint_B1"):
        model.infer_joint_B1 = model.infer_joint
    model.infer_joint_batch = MethodType(infer_joint_batch, model)
