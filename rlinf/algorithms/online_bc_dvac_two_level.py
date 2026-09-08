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

"""Detached DVAC factors for one complete sampled online-BC optimizer batch.

The caller must pass the complete single-rank batch before microbatch splitting.
This module does not sample replay, gather ranks, infer scenes, or alter masks.
Collected variance is fixed; the outer factor is recomputed for every batch.
"""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def compute_two_level_bc_weights(
    variance: torch.Tensor,
    action_valid_mask: torch.Tensor,
    alpha_local: float = 1.0,
    alpha_chunk: float = 1.0,
    variance_eps: float = 1e-12,
    range_eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return detached weights ``[B,H]`` and finite scalar diagnostics.

    ``variance[B,H]`` is the collected nonnegative endpoint variance.
    ``action_valid_mask[B,H,D]`` is the exact binary FM supervision mask; a
    ``[B,H]`` mask is equivalent to D=1. Padding does not enter any statistics
    and receives weight one. Missing or invalid variance at supervised actions
    raises even when both strengths are zero.

    Local MinMax(log V) is centered with the number of valid D elements per H.
    Its factor L therefore has mean one under each query's actual masked FM
    reduction. The outer score is the raw masked mean(log V), BEFORE local
    normalization. Its MinMax is centered equally over all valid sampled
    queries, counting repeated replay draws with their actual multiplicity.
    There is no final per-query normalization to erase the outer factor g.

    Both strengths lie in [0,1]; constants produce factor one. Computation
    promotes low precision to float32 and retains float64 input precision.
    An empty B or an all-invalid mask returns neutral weights and zero counts;
    the caller retains the original BC skip/zero-loss behavior.

    ``weight_mean/std`` and ``local_std`` use the BC reduction: masked HD
    average within each query, then equal average over valid queries.
    ``weight_sumsq`` sums W**2 over valid HD elements. Coefficient diagnostics
    instead use c_bhd = W_bh / n_b on valid elements (the common 1/B is omitted,
    which leaves ESS unchanged); ESS=(sum c)**2/sum(c**2). ``chunk_std`` is the
    population std of g over valid queries. These measure coefficients, not
    gradient norms. Invalid/empty-domain means and extrema use neutral one.
    """
    if variance.ndim != 2 or variance.shape[1] == 0:
        raise ValueError("variance must have shape [B,H] with H > 0")
    if not variance.is_floating_point():
        raise TypeError("variance must be a floating-point tensor")
    if action_valid_mask.ndim not in (2, 3):
        raise ValueError("action_valid_mask must have shape [B,H] or [B,H,D]")
    if action_valid_mask.shape[:2] != variance.shape:
        raise ValueError("action_valid_mask and variance must align in [B,H]")
    if action_valid_mask.ndim == 3 and action_valid_mask.shape[2] == 0:
        raise ValueError("action_valid_mask must have D > 0")
    for name, value in (("alpha_local", alpha_local), ("alpha_chunk", alpha_chunk)):
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be finite and in [0,1]")
    for name, value in (("variance_eps", variance_eps), ("range_eps", range_eps)):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    dtype = torch.float64 if variance.dtype == torch.float64 else torch.float32
    values = variance.detach().to(dtype=dtype)
    mask = action_valid_mask.detach().to(device=values.device)
    if mask.is_complex() or not torch.isfinite(mask).all():
        raise ValueError("action_valid_mask must be finite and binary")
    if not ((mask == 0) | (mask == 1)).all():
        raise ValueError("action_valid_mask must be finite and binary")
    for name, value in (("variance_eps", variance_eps), ("range_eps", range_eps)):
        represented = torch.as_tensor(value, device=values.device, dtype=dtype)
        if not torch.isfinite(represented) or represented <= 0:
            raise ValueError(
                f"{name} must be representable as a finite positive {dtype}"
            )

    q = mask.to(dtype=dtype)
    if q.ndim == 3:
        q = q.sum(dim=-1)
    valid_h = q > 0
    n = q.sum(dim=-1)
    valid_b = n > 0
    if not torch.isfinite(values[valid_h]).all() or (values[valid_h] < 0).any():
        raise ValueError("supervised DVAC variance must be finite and non-negative")

    diagnostics = {
        "weight_mean": 1.0,
        "weight_min": 1.0,
        "weight_max": 1.0,
        "weight_std": 0.0,
        "weight_sumsq": 0.0,
        "local_std": 0.0,
        "chunk_std": 0.0,
        "coefficient_sum": 0.0,
        "coefficient_sumsq": 0.0,
        "coefficient_ess": 0.0,
        "coefficient_ess_fraction": 0.0,
        "valid_query_count": float(valid_b.sum().item()),
        "valid_action_count": float(valid_h.sum().item()),
        "valid_element_count": float(q.sum().item()),
        "total_query_count": float(variance.shape[0]),
    }
    if not valid_b.any():
        return torch.ones_like(values), diagnostics

    safe_values = torch.where(valid_h, values, torch.zeros_like(values))
    log_v = torch.log(safe_values + float(variance_eps))
    if not torch.isfinite(log_v[valid_h]).all():
        raise ValueError("supervised log variance is non-finite; check variance_eps")
    log_v = torch.where(valid_h, log_v, torch.zeros_like(log_v))
    minimum = log_v.masked_fill(~valid_h, float("inf")).amin(dim=-1)
    maximum = log_v.masked_fill(~valid_h, -float("inf")).amax(dim=-1)
    minimum = torch.where(valid_b, minimum, torch.zeros_like(minimum))
    maximum = torch.where(valid_b, maximum, torch.zeros_like(maximum))
    unit = (log_v - minimum[:, None]) / (
        maximum[:, None] - minimum[:, None] + float(range_eps)
    )
    unit = torch.where(valid_h, unit, torch.zeros_like(unit))
    denominator = n.clamp_min(1.0)
    center = (unit * q).sum(dim=-1) / denominator
    local = 1.0 + float(alpha_local) * (unit - center[:, None])
    local = torch.where(valid_h, local, torch.ones_like(local))

    raw_score = (log_v * q).sum(dim=-1) / denominator
    scores = raw_score[valid_b]
    outer_unit = (scores - scores.min()) / (
        scores.max() - scores.min() + float(range_eps)
    )
    chunk = torch.ones_like(n)
    chunk[valid_b] = 1.0 + float(alpha_chunk) * (outer_unit - outer_unit.mean())
    weights = torch.where(valid_h, local * chunk[:, None], torch.ones_like(local))

    # Every valid query contributes equal total mass to the original BC loss.
    mean = ((weights * q).sum(dim=-1) / denominator)[valid_b].mean()
    second = (
        (((weights - mean).square() * q).sum(dim=-1) / denominator)[valid_b].mean()
    )
    local_second = (
        (((local - 1.0).square() * q).sum(dim=-1) / denominator)[valid_b].mean()
    )
    coefficient_sum = (weights * q / denominator[:, None]).sum()
    coefficient_sumsq = (
        weights.square() * q / denominator[:, None].square()
    ).sum()
    ess = coefficient_sum.square() / coefficient_sumsq
    diagnostics.update(
        weight_mean=float(mean.item()),
        weight_min=float(weights[valid_h].min().item()),
        weight_max=float(weights[valid_h].max().item()),
        weight_std=float(second.sqrt().item()),
        weight_sumsq=float((weights.square() * q).sum().item()),
        local_std=float(local_second.sqrt().item()),
        chunk_std=float(chunk[valid_b].std(unbiased=False).item()),
        coefficient_sum=float(coefficient_sum.item()),
        coefficient_sumsq=float(coefficient_sumsq.item()),
        coefficient_ess=float(ess.item()),
        coefficient_ess_fraction=float((ess / q.sum()).item()),
    )
    return weights, diagnostics
