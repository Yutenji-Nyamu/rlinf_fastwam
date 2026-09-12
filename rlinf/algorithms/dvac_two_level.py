# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Detached action/chunk DVAC factors for explicit scene groups.

Call once on the gathered rollout tensors, before optimization shuffling. This
module does not gather ranks, infer group IDs, change advantages, or change the
actor loss mask/reduction. Both mappings increase with log variance, matching
the direction of the previous action-advantage DVAC implementation.
"""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def compute_dvac_two_level_weights(
    variance: torch.Tensor,
    loss_mask: torch.Tensor,
    group_ids: torch.Tensor,
    advantages: torch.Tensor,
    *,
    scope: str = "both",
    alpha_local: float = 1.0,
    alpha_chunk: float = 1.0,
    log_eps: float = 1e-12,
    minmax_eps: float = 1e-6,
    chunk_contributions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return action weights ``[T,B,H]`` and detached diagnostic tensors.

    ``loss_mask`` is the actual binary *chunk* mask, shaped ``[T,B]`` or
    ``[T,B,1]``. Every H position of an actor-valid chunk participates; an
    executed-action mask is deliberately not accepted. ``group_ids[B]`` contains
    stable integer scene-group IDs assigned before shuffling. ``advantages``
    has shape ``[T,B,1]`` and is used only to select the scope.

    For ``both``, all valid chunks with positive contribution participate,
    including zero-advantage chunks. ``positive``/``negative`` select strictly
    A>0/A<0, respectively, and compute outer statistics only in that selected
    set. Excluded chunks have both factors and final weights equal to one.

    The inner factor is 1 + alpha_local * (MinMax(log V) - mean_H).
    The outer score is the raw mean_H(log V), before inner normalization.
    Within each scene group, MinMax outer scores are centered using the
    optional nonnegative ``chunk_contributions[T,B]`` from the actor's loss
    aggregation. Zero-contribution chunks do not affect either normalization.
    No history, sigmoid, extra clipping, or final chunk renormalization is used.

    Computation uses float32, retaining float64 when variance is float64.
    Diagnostics: local_factors/log_variance [T,B,H]; chunk_factors,
    chunk_scores, valid_mask, eligible_mask, chunk_contributions [T,B,1];
    group_ids [B]. Invalid log-variance/scores use zero as a masked sentinel.
    """
    if variance.ndim != 3 or variance.shape[-1] == 0:
        raise ValueError("variance must have shape [T,B,H] with H > 0")
    if not variance.is_floating_point():
        raise TypeError("variance must be a floating-point tensor")
    time_steps, batch_size, _ = variance.shape
    chunk_shape = (time_steps, batch_size)
    if loss_mask.shape not in (chunk_shape, (*chunk_shape, 1)):
        raise ValueError("loss_mask must be a chunk mask [T,B] or [T,B,1]")
    if advantages.shape != (*chunk_shape, 1):
        raise ValueError("advantages must have shape [T,B,1]")
    if not advantages.is_floating_point():
        raise TypeError("advantages must be a floating-point tensor")
    if group_ids.shape != (batch_size,):
        raise ValueError("group_ids must have shape [B]")
    if group_ids.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise TypeError("group_ids must contain integer IDs, not inferred positions")
    if scope not in ("both", "positive", "negative"):
        raise ValueError("scope must be 'both', 'positive', or 'negative'")
    for name, value in (("alpha_local", alpha_local), ("alpha_chunk", alpha_chunk)):
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be finite and in [0,1]")
    for name, value in (("log_eps", log_eps), ("minmax_eps", minmax_eps)):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    device = variance.device
    dtype = torch.float64 if variance.dtype == torch.float64 else torch.float32
    values = variance.detach().to(dtype=dtype)
    mask_values = loss_mask.detach().to(device=device).reshape(chunk_shape)
    if mask_values.is_complex() or not torch.isfinite(mask_values).all():
        raise ValueError("loss_mask must be finite and binary")
    if not ((mask_values == 0) | (mask_values == 1)).all():
        raise ValueError("loss_mask must be finite and binary")
    valid = mask_values.bool()
    valid_h = valid.unsqueeze(-1).expand_as(values)
    if not torch.isfinite(values[valid_h]).all() or (values[valid_h] < 0).any():
        raise ValueError("actor-valid DVAC variance must be finite and non-negative")

    adv = advantages.detach().to(device=device, dtype=dtype).squeeze(-1)
    if not torch.isfinite(adv[valid]).all():
        raise ValueError("actor-valid advantages must be finite")
    ids = group_ids.detach().to(device=device, dtype=torch.int64)
    if chunk_contributions is None:
        contributions = torch.ones(chunk_shape, device=device, dtype=dtype)
    else:
        if chunk_contributions.shape != chunk_shape:
            raise ValueError("chunk_contributions must have shape [T,B]")
        if chunk_contributions.is_complex():
            raise ValueError("chunk_contributions must be real and non-negative")
        contributions = chunk_contributions.detach().to(device=device, dtype=dtype)
        if (
            not torch.isfinite(contributions[valid]).all()
            or (contributions[valid] < 0).any()
        ):
            raise ValueError(
                "actor-valid chunk_contributions must be finite and non-negative"
            )
    contributions = torch.where(valid, contributions, torch.zeros_like(contributions))
    eligible = valid & (contributions > 0)
    if scope == "positive":
        eligible = eligible & (adv > 0)
    elif scope == "negative":
        eligible = eligible & (adv < 0)

    safe_values = torch.where(valid_h, values, torch.zeros_like(values))
    log_variance = torch.log(safe_values + float(log_eps))
    if not torch.isfinite(log_variance[valid_h]).all():
        raise ValueError(
            "actor-valid log variance is non-finite; check log_eps and dtype"
        )
    log_variance = torch.where(valid_h, log_variance, torch.zeros_like(log_variance))
    chunk_scores = log_variance.mean(dim=-1)

    local_min = log_variance.amin(dim=-1, keepdim=True)
    local_span = log_variance.amax(dim=-1, keepdim=True) - local_min
    local_unit = (log_variance - local_min) / (local_span + float(minmax_eps))
    local_factors = 1.0 + float(alpha_local) * (
        local_unit - local_unit.mean(dim=-1, keepdim=True)
    )
    local_factors = torch.where(
        eligible.unsqueeze(-1), local_factors, torch.ones_like(local_factors)
    )

    chunk_factors = torch.ones(chunk_shape, device=device, dtype=dtype)
    for group_id in torch.unique(ids[eligible.any(dim=0)]):
        selected = eligible & (ids == group_id).unsqueeze(0)
        scores = chunk_scores[selected]
        unit = (scores - scores.min()) / (
            scores.max() - scores.min() + float(minmax_eps)
        )
        mass = contributions[selected]
        # Rescaling mass leaves the weighted center unchanged and avoids an
        # overflowing sum when the actor contribution coefficients are large.
        mass = mass / mass.max()
        center = (unit * mass).sum() / mass.sum()
        chunk_factors[selected] = 1.0 + float(alpha_chunk) * (unit - center)

    weights = local_factors * chunk_factors.unsqueeze(-1)
    return weights, {
        "local_factors": local_factors,
        "chunk_factors": chunk_factors.unsqueeze(-1),
        "chunk_scores": chunk_scores.unsqueeze(-1),
        "log_variance": log_variance,
        "valid_mask": valid.unsqueeze(-1),
        "eligible_mask": eligible.unsqueeze(-1),
        "chunk_contributions": contributions.unsqueeze(-1),
        "group_ids": ids,
    }
