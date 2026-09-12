# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Two-level reference-BC weights for one complete RLT training batch."""

from __future__ import annotations

import math

import torch


def _centered_minmax(
    values: torch.Tensor, *, alpha: float, eps: float
) -> torch.Tensor:
    """Return mean-one factors along the last axis, neutral on constant domains."""
    low = values.amin(dim=-1, keepdim=True)
    span = values.amax(dim=-1, keepdim=True) - low
    scaled = torch.where(
        span > eps,
        (values - low) / span.clamp_min(eps),
        torch.zeros_like(values),
    )
    return 1.0 + alpha * (scaled - scaled.mean(dim=-1, keepdim=True))


def _weight_moments(weights: torch.Tensor) -> dict[str, float]:
    """Summarize flattened factors; empty domains have neutral diagnostics."""
    flat = weights.detach().double().reshape(-1)
    if flat.numel() == 0:
        return {"mean": 1.0, "std": 0.0, "ess_ratio": 1.0}
    # Rescaling before squaring keeps ESS stable for a large success_scale.
    scaled = flat / flat.amax().clamp_min(torch.finfo(flat.dtype).tiny)
    ess = scaled.sum().square() / (
        flat.numel() * scaled.square().sum()
    ).clamp_min(torch.finfo(flat.dtype).tiny)
    return {
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "ess_ratio": float(ess.item()),
    }


def build_two_level_success_weights(
    variances: torch.Tensor,
    episode_success: torch.Tensor,
    *,
    alpha_local: float = 1.0,
    alpha_chunk: float = 1.0,
    log_eps: float = 1e-12,
    minmax_eps: float = 1e-6,
    success_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Build two-level weights before splitting a complete batch into microbatches.

    Each successful query gets a mean-one factor over its H action positions.
    Its unnormalized mean log-variance supplies a second factor, centered over
    successful queries in this sampled batch. Failed queries never participate
    in either factor's comparison and keep weight one. Duplicate replay samples
    count with their actual multiplicity in the sampled batch.

    Args:
        variances: Floating tensor [B, H] of cached teacher variances. All rows,
            including failures, must be finite and nonnegative. H must be positive.
        episode_success: Boolean tensor [B]; no implicit truth-value conversion.
        alpha_local: Within-query allocation strength in [0, 1].
        alpha_chunk: Between-successful-query allocation strength in [0, 1].
        log_eps: Positive finite offset in log(V + eps).
        minmax_eps: Positive finite range threshold for returning neutral factors.
        success_scale: Positive finite multiplier for successful queries only.

    Returns:
        Detached float32 weights [B, H] on the input device, and complete-batch
        diagnostics. Inner/outer diagnostics concern successes only; applied
        diagnostics include failed queries. Empty domains report neutral moments.

    Raises:
        ValueError: Inputs, parameters, or resulting float32 weights are invalid.
    """
    if (
        not isinstance(variances, torch.Tensor)
        or variances.ndim != 2
        or variances.shape[1] == 0
        or not torch.is_floating_point(variances)
    ):
        raise ValueError("RLT two-level variances must be floating [B,H] with H > 0.")
    if (
        not isinstance(episode_success, torch.Tensor)
        or episode_success.dtype != torch.bool
        or episode_success.ndim != 1
        or episode_success.shape[0] != variances.shape[0]
    ):
        raise ValueError("RLT two-level episode_success must be bool [B].")

    parameters = {
        "alpha_local": float(alpha_local),
        "alpha_chunk": float(alpha_chunk),
        "log_eps": float(log_eps),
        "minmax_eps": float(minmax_eps),
        "success_scale": float(success_scale),
    }
    for name, value in parameters.items():
        if not math.isfinite(value):
            raise ValueError(f"RLT two-level {name} must be finite.")
        if name.startswith("alpha_"):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"RLT two-level {name} must be in [0,1].")
        elif value <= 0.0:
            raise ValueError(f"RLT two-level {name} must be positive.")
    if not torch.isfinite(variances).all() or (variances < 0).any():
        raise ValueError("RLT two-level variances must all be finite and nonnegative.")

    success = episode_success.detach().to(device=variances.device)
    batch_size, horizon = variances.shape
    success_count = int(success.sum().item())
    weights = torch.ones_like(variances, dtype=torch.float32)
    inner = torch.empty((0, horizon), dtype=torch.float64, device=variances.device)
    outer = torch.empty(0, dtype=torch.float64, device=variances.device)
    if success_count:
        values = variances.detach()[success].double()
        # Stable log(V + eps), including V=0 and very small positive eps. The
        # tiny B x H reduction does not involve either policy or teacher graphs.
        log_values = torch.logaddexp(
            values.log(), values.new_tensor(math.log(parameters["log_eps"]))
        )
        inner = _centered_minmax(
            log_values,
            alpha=parameters["alpha_local"],
            eps=parameters["minmax_eps"],
        )
        chunk_signal = log_values.mean(dim=-1)
        outer = _centered_minmax(
            chunk_signal.unsqueeze(0),
            alpha=parameters["alpha_chunk"],
            eps=parameters["minmax_eps"],
        ).squeeze(0)
        success_weights = (
            inner * outer[:, None] * parameters["success_scale"]
        ).float()
        if not torch.isfinite(success_weights).all() or (success_weights <= 0).any():
            raise ValueError(
                "RLT two-level success weights must be positive finite float32; "
                "success_scale may exceed the representable range."
            )
        weights[success] = success_weights

    prefix = "rlt_dvac_new/"
    metrics = {
        f"{prefix}success_count": float(success_count),
        f"{prefix}success_ratio": success_count / batch_size if batch_size else 0.0,
        f"{prefix}outer_min": float(outer.min().item()) if success_count else 1.0,
        f"{prefix}outer_max": float(outer.max().item()) if success_count else 1.0,
    }
    for name, factors in (
        ("inner", inner),
        ("outer", outer),
        ("success_applied", weights[success]),
        ("applied", weights),
    ):
        metrics.update(
            {
                f"{prefix}{name}_{key}": value
                for key, value in _weight_moments(factors).items()
            }
        )
    return weights.detach(), metrics
