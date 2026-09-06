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

"""Trajectory-level DVAC quality used by Prism-style RLOO."""

from __future__ import annotations

import torch


def trajectory_mean_log_variance(
    variance: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    log_eps: float,
) -> torch.Tensor:
    """Average ``log(V_L + eps)`` over actions actually executed per trajectory."""

    if variance.ndim != 3 or action_mask.shape != variance.shape:
        raise ValueError(
            "variance and action_mask must both have shape [T,B,H], got "
            f"{tuple(variance.shape)} and {tuple(action_mask.shape)}"
        )
    if log_eps <= 0:
        raise ValueError("log_eps must be positive")
    if not torch.isfinite(variance).all() or (variance < 0).any():
        raise ValueError("DVAC variance must be finite and non-negative")

    mask = action_mask.to(device=variance.device, dtype=torch.bool)
    counts = mask.sum(dim=(0, 2))
    if (counts == 0).any():
        raise ValueError("each trajectory must contain at least one executed action")
    log_variance = torch.log(variance.float() + float(log_eps))
    return (log_variance * mask).sum(dim=(0, 2)) / counts


def reverse_rank_quality(cost: torch.Tensor, group_size: int) -> torch.Tensor:
    """Map lower costs to higher qualities in each group, with average-rank ties."""

    if cost.ndim != 1:
        raise ValueError(f"cost must have shape [B], got {tuple(cost.shape)}")
    if group_size < 2 or cost.numel() % group_size != 0:
        raise ValueError(
            f"batch {cost.numel()} must be divisible by group_size={group_size} >= 2"
        )
    if not torch.isfinite(cost).all():
        raise ValueError("trajectory DVAC cost contains NaN or Inf")

    grouped = cost.reshape(-1, group_size)
    pairwise = grouped.unsqueeze(-1) - grouped.unsqueeze(-2)
    lower_count = (pairwise > 0).sum(dim=-1)
    equal_count = (pairwise == 0).sum(dim=-1)
    average_rank = lower_count + 0.5 * (equal_count - 1)
    quality = 1.0 - average_rank.to(grouped.dtype) / float(group_size - 1)
    return quality.reshape_as(cost)


def trajectory_dvac_quality(
    variance: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    group_size: int,
    log_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return trajectory cost and its within-group reverse-rank quality."""

    cost = trajectory_mean_log_variance(variance, action_mask, log_eps=log_eps)
    return cost, reverse_rank_quality(cost, group_size)
