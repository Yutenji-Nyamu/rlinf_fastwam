# Copyright 2026 The RLinf Authors.
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

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch


def compute_endpoint_variances(
    endpoint_previews: torch.Tensor,
    l_values: Iterable[int] = (2, 3, 4),
) -> dict[int, torch.Tensor]:
    """Return action-dimension-summed population variance for each denoise tail."""

    if endpoint_previews.ndim != 4:
        raise ValueError(
            "endpoint_previews must be [B,M,H,D], got "
            f"{tuple(endpoint_previews.shape)}"
        )
    num_steps = int(endpoint_previews.shape[1])
    variances = {}
    for l_value in tuple(int(value) for value in l_values):
        if l_value < 2 or l_value > num_steps:
            raise ValueError(f"DVAC tail L={l_value} is invalid for M={num_steps}.")
        variances[l_value] = (
            endpoint_previews[:, -l_value:]
            .float()
            .var(dim=1, unbiased=False)
            .sum(dim=-1)
        )
    return variances


@dataclass
class FrozenGlobalZMoments:
    """Streaming log-DVAC moments, frozen when replay warm-up completes."""

    log_eps: float = 1e-12
    std_floor: float = 1e-6
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    frozen: bool = False
    mean: float = 0.0
    std: float = 1.0

    def update_variances(self, variances: torch.Tensor) -> None:
        if self.frozen:
            return
        values = torch.log(
            variances.detach().float().clamp_min(0) + float(self.log_eps)
        ).double()
        values = values.reshape(-1)
        self.count += int(values.numel())
        self.total += float(values.sum().item())
        self.total_sq += float(values.square().sum().item())

    def sufficient_statistics(self) -> tuple[int, float, float]:
        return self.count, self.total, self.total_sq

    def freeze_from_statistics(
        self, count: int, total: float, total_sq: float
    ) -> None:
        if int(count) <= 0:
            raise ValueError("Cannot freeze an empty RLT DVAC baseline.")
        mean = float(total) / int(count)
        variance = max(float(total_sq) / int(count) - mean * mean, 0.0)
        self.count = int(count)
        self.total = float(total)
        self.total_sq = float(total_sq)
        self.mean = mean
        self.std = max(math.sqrt(variance), float(self.std_floor))
        self.frozen = True

    def state_dict(self) -> dict[str, Any]:
        return {
            "log_eps": float(self.log_eps),
            "std_floor": float(self.std_floor),
            "count": int(self.count),
            "total": float(self.total),
            "total_sq": float(self.total_sq),
            "frozen": bool(self.frozen),
            "mean": float(self.mean),
            "std": float(self.std),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        required = {
            "log_eps",
            "std_floor",
            "count",
            "total",
            "total_sq",
            "frozen",
            "mean",
            "std",
        }
        missing = sorted(required.difference(state))
        if missing:
            raise ValueError(f"RLT DVAC baseline is missing keys: {missing}.")
        self.log_eps = float(state["log_eps"])
        self.std_floor = float(state["std_floor"])
        self.count = int(state["count"])
        self.total = float(state["total"])
        self.total_sq = float(state["total_sq"])
        self.frozen = bool(state["frozen"])
        self.mean = float(state["mean"])
        self.std = float(state["std"])


def global_z_scores(
    variances: torch.Tensor,
    *,
    mean: float,
    std: float,
    log_eps: float = 1e-12,
    std_floor: float = 1e-6,
    z_clip: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standardize log-DVAC with the frozen global baseline."""

    log_variances = torch.log(
        variances.float().clamp_min(0) + float(log_eps)
    )
    z_scores = (
        (log_variances - float(mean)) / max(float(std), float(std_floor))
    ).clamp(-float(z_clip), float(z_clip))
    return z_scores.detach(), log_variances.detach()


def centered_mean_one_weights(
    z_scores: torch.Tensor, *, strength: float
) -> torch.Tensor:
    """Apply AutoDL Pure's non-negative, per-query mean-one C-horizon mapping."""

    if z_scores.ndim != 2:
        raise ValueError(
            f"RLT DVAC z_scores must be [B,H], got {tuple(z_scores.shape)}."
        )
    if float(strength) < 0:
        raise ValueError("RLT DVAC centered strength must be non-negative.")
    centered = z_scores.float() - z_scores.float().mean(dim=-1, keepdim=True)
    weights = (1.0 + float(strength) * centered).clamp_min(0.0)
    return (weights / weights.mean(dim=-1, keepdim=True).clamp_min(1e-12)).detach()


def episode_success_flags(rewards: torch.Tensor) -> torch.Tensor:
    """Return one success flag per environment from a complete [T,B,...] rollout."""

    if rewards.ndim < 2:
        raise ValueError(
            f"RLT episode rewards must be [T,B,...], got {tuple(rewards.shape)}."
        )
    by_step_env = rewards.detach().float().reshape(
        rewards.shape[0], rewards.shape[1], -1
    )
    return (by_step_env > 0).any(dim=0).any(dim=-1)


def build_rlt_bc_targets_and_weights(
    executed_actions: torch.Tensor,
    reference_actions: torch.Tensor,
    human_mask: torch.Tensor,
    *,
    episode_success: torch.Tensor | None = None,
    success_weights: torch.Tensor | None = None,
    apply_success_weights: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep reference BC targets and reweight every position in successful rows."""

    if executed_actions.shape != reference_actions.shape or executed_actions.ndim != 3:
        raise ValueError(
            "RLT BC actions must be matching [B,H,D] tensors, got "
            f"{tuple(executed_actions.shape)} and {tuple(reference_actions.shape)}."
        )
    if tuple(human_mask.shape) != tuple(executed_actions.shape[:2]):
        raise ValueError(
            "RLT BC human mask/action shape mismatch: "
            f"{tuple(human_mask.shape)} vs {tuple(executed_actions.shape)}."
        )

    targets = torch.where(human_mask[..., None], executed_actions, reference_actions)
    weights = torch.ones_like(human_mask, dtype=executed_actions.dtype)
    success_mask = torch.zeros_like(human_mask, dtype=torch.bool)
    if apply_success_weights:
        if episode_success is None or success_weights is None:
            raise ValueError(
                "Successful-episode RLT DVAC BC requires success flags and weights."
            )
        success_query = (
            episode_success.detach().to(human_mask.device).bool().reshape(-1)
        )
        if success_query.shape[0] != executed_actions.shape[0]:
            raise ValueError("RLT BC episode-success/action batch mismatch.")
        if tuple(success_weights.shape) != tuple(executed_actions.shape[:2]):
            raise ValueError("RLT BC success-weight/action shape mismatch.")
        success_mask = success_query[:, None].expand_as(human_mask)
        weights = torch.where(
            success_mask,
            success_weights.detach().to(device=weights.device, dtype=weights.dtype),
            weights,
        )
    return targets, weights.detach(), success_mask


def summarize_weights(
    weights: torch.Tensor,
    z_scores: torch.Tensor,
    log_variances: torch.Tensor,
) -> dict[str, float]:
    """Return the small metric set needed to verify Pure04 is active."""

    flat_w = weights.detach().float().reshape(-1)
    quantiles = torch.quantile(
        flat_w, torch.tensor([0.05, 0.5, 0.95], device=flat_w.device)
    )
    horizon = max(int(weights.shape[-1]), 1)
    sums = weights.float().sum(dim=-1)
    ess = sums.square() / (
        horizon * weights.float().square().sum(dim=-1)
    ).clamp_min(1e-12)
    return {
        "rlt_dvac/log_v_mean": float(log_variances.float().mean().item()),
        "rlt_dvac/z_std": float(z_scores.float().std(unbiased=False).item()),
        "rlt_dvac/weight_p05": float(quantiles[0].item()),
        "rlt_dvac/weight_median": float(quantiles[1].item()),
        "rlt_dvac/weight_mean": float(flat_w.mean().item()),
        "rlt_dvac/weight_p95": float(quantiles[2].item()),
        "rlt_dvac/weight_ess_ratio": float(ess.mean().item()),
    }
