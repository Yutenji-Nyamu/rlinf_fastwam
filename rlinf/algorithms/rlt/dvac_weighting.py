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
    """Compute DVAC population variance over the last L endpoint previews.

    Args:
        endpoint_previews: ``[B, M, H, D_active]`` clean-action previews.
        l_values: Tail lengths to summarize.

    Returns:
        A mapping from ``L`` to ``[B, H]`` variance summed over action dims.
    """

    if endpoint_previews.ndim != 4:
        raise ValueError(
            f"endpoint_previews must be [B,M,H,D], got {tuple(endpoint_previews.shape)}"
        )
    num_steps = int(endpoint_previews.shape[1])
    variances: dict[int, torch.Tensor] = {}
    for l_value in tuple(int(value) for value in l_values):
        if l_value < 2 or l_value > num_steps:
            raise ValueError(f"DVAC tail L={l_value} is invalid for M={num_steps}.")
        tail = endpoint_previews[:, -l_value:].float()
        variances[l_value] = tail.var(dim=1, unbiased=False).sum(dim=-1)
    return variances


@dataclass
class FrozenGlobalZMoments:
    """Streaming log-V moments that freeze once the RLT replay is ready."""

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
        values = torch.log(variances.detach().float().clamp_min(0) + self.log_eps)
        values = values.double().reshape(-1)
        self.count += int(values.numel())
        self.total += float(values.sum().item())
        self.total_sq += float(values.square().sum().item())

    def sufficient_statistics(self) -> tuple[int, float, float]:
        return self.count, self.total, self.total_sq

    def freeze_from_statistics(
        self,
        count: int,
        total: float,
        total_sq: float,
    ) -> None:
        if count <= 0:
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
        self.log_eps = float(state["log_eps"])
        self.std_floor = float(state["std_floor"])
        self.count = int(state["count"])
        self.total = float(state["total"])
        self.total_sq = float(state["total_sq"])
        self.frozen = bool(state["frozen"])
        self.mean = float(state["mean"])
        self.std = float(state["std"])


def global_z_weights(
    variances: torch.Tensor,
    *,
    mean: float,
    std: float,
    log_eps: float = 1e-12,
    std_floor: float = 1e-6,
    z_clip: float = 2.0,
    strength: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map teacher DVAC to detached per-future-action weights."""

    log_variances = torch.log(variances.float().clamp_min(0) + log_eps)
    z_scores = ((log_variances - float(mean)) / max(float(std), std_floor)).clamp(
        -float(z_clip), float(z_clip)
    )
    weights = 1.0 + float(strength) * z_scores
    return weights.detach(), z_scores.detach(), log_variances.detach()


def straight_through_scale_actions(
    actions: torch.Tensor,
    weights: torch.Tensor,
    *,
    action_dim: int,
) -> torch.Tensor:
    """Keep actor actions unchanged forward and scale Q gradients per h."""

    action_chunk = actions.reshape(actions.shape[0], -1, int(action_dim))
    if tuple(weights.shape) != tuple(action_chunk.shape[:2]):
        raise ValueError(
            "RLT DVAC weights/action shape mismatch: "
            f"weights={tuple(weights.shape)}, actions={tuple(action_chunk.shape)}"
        )
    scaled = action_chunk.detach() + weights.detach().unsqueeze(-1) * (
        action_chunk - action_chunk.detach()
    )
    return scaled.reshape_as(actions)


def summarize_weights(
    weights: torch.Tensor,
    z_scores: torch.Tensor,
    log_variances: torch.Tensor,
    *,
    weight_min: float,
    weight_max: float,
) -> dict[str, float]:
    """Small diagnostics for the RLT training metric stream."""

    flat_w = weights.detach().float().reshape(-1)
    flat_z = z_scores.detach().float().reshape(-1)
    flat_log_v = log_variances.detach().float().reshape(-1)
    quantiles = torch.quantile(
        flat_w, torch.tensor([0.05, 0.5, 0.95], device=flat_w.device)
    )

    per_query_sum = weights.float().sum(dim=-1)
    per_query_sum_sq = weights.float().square().sum(dim=-1)
    horizon = max(int(weights.shape[-1]), 1)
    ess = per_query_sum.square() / (horizon * per_query_sum_sq).clamp_min(1e-12)

    top_k = max(int(math.ceil(0.2 * horizon)), 1)
    top_mass = weights.float().topk(top_k, dim=-1).values.sum(dim=-1)
    total_mass = weights.float().sum(dim=-1).clamp_min(1e-12)

    return {
        "rlt_dvac/log_v_mean": float(flat_log_v.mean().item()),
        "rlt_dvac/log_v_std": float(flat_log_v.std(unbiased=False).item()),
        "rlt_dvac/z_mean": float(flat_z.mean().item()),
        "rlt_dvac/z_std": float(flat_z.std(unbiased=False).item()),
        "rlt_dvac/weight_p05": float(quantiles[0].item()),
        "rlt_dvac/weight_median": float(quantiles[1].item()),
        "rlt_dvac/weight_mean": float(flat_w.mean().item()),
        "rlt_dvac/weight_p95": float(quantiles[2].item()),
        "rlt_dvac/downweighted_fraction": float((flat_w < 1.0).float().mean().item()),
        "rlt_dvac/upweighted_fraction": float((flat_w > 1.0).float().mean().item()),
        "rlt_dvac/min_weight_fraction": float(
            (flat_w <= float(weight_min) + 1e-6).float().mean().item()
        ),
        "rlt_dvac/max_weight_fraction": float(
            (flat_w >= float(weight_max) - 1e-6).float().mean().item()
        ),
        "rlt_dvac/weight_ess_ratio": float(ess.mean().item()),
        "rlt_dvac/top20_weight_mass": float((top_mass / total_mass).mean().item()),
    }
