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

"""DVAC-based per-action gradient weighting for embodied policy optimization.

Rollout workers compute one endpoint-variance tensor. Actor workers normalize
its log values against completed runner steps and use straight-through scaling
to change gradients without changing the PPO/GRPO forward values.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import torch


def compute_endpoint_variance(
    z_endpoint: torch.Tensor,
    tail_steps: int,
) -> torch.Tensor:
    """Return population endpoint variance ``[B,H]`` for the final L previews."""

    if z_endpoint.ndim != 4:
        raise ValueError(
            "z_endpoint must have shape [B,M,H,D_active], got "
            f"{tuple(z_endpoint.shape)}"
        )
    tail_steps = int(tail_steps)
    if tail_steps < 2 or tail_steps > z_endpoint.shape[1]:
        raise ValueError(
            f"tail_steps={tail_steps} is invalid for M={z_endpoint.shape[1]}"
        )
    z = z_endpoint.detach().float()
    if not torch.isfinite(z).all():
        raise ValueError("z_endpoint contains NaN or Inf")
    return z[:, -tail_steps:].var(dim=1, unbiased=False).sum(dim=-1)


def straight_through_scale_logprobs(
    logprobs: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Keep log-prob values unchanged while scaling each future-h gradient."""

    if logprobs.ndim != 3:
        raise ValueError(
            f"logprobs must have shape [B,H,D], got {tuple(logprobs.shape)}"
        )
    if weights.shape != logprobs.shape[:2]:
        raise ValueError(
            "weights must have shape [B,H], got "
            f"{tuple(weights.shape)} for {tuple(logprobs.shape)}"
        )
    if not torch.isfinite(weights).all():
        raise ValueError("DVAC weights contain NaN or Inf")
    detached = logprobs.detach()
    scale = weights.detach().to(device=logprobs.device, dtype=logprobs.dtype)
    return detached + scale.unsqueeze(-1) * (logprobs - detached)


@dataclass(frozen=True)
class DVACStepStats:
    runner_step: int
    count: int
    value_sum: float
    value_sq_sum: float

    @property
    def mean(self) -> float:
        return self.value_sum / self.count if self.count else 0.0

    @property
    def std(self) -> float:
        if not self.count:
            return 0.0
        variance = max(self.value_sq_sum / self.count - self.mean**2, 0.0)
        return math.sqrt(variance)


def local_log_v_sufficient_statistics(
    variance: torch.Tensor,
    *,
    log_eps: float,
) -> torch.Tensor:
    """Return local ``count,sum,sumsq`` over all query/future positions."""

    if variance.ndim != 3:
        raise ValueError(
            f"variance must have shape [T,B,H], got {tuple(variance.shape)}"
        )
    if not torch.isfinite(variance).all() or (variance < 0).any():
        raise ValueError("DVAC variance must be finite and non-negative")
    log_v = torch.log(variance.double() + float(log_eps)).reshape(-1)
    return torch.stack(
        (
            torch.tensor(
                float(log_v.numel()), dtype=torch.float64, device=variance.device
            ),
            log_v.sum(),
            log_v.square().sum(),
        )
    )


class DVACRecentStats:
    """Rolling statistics from completed runner steps only."""

    def __init__(
        self,
        *,
        window_steps: int,
        warmup_steps: int,
        log_eps: float,
        std_floor: float,
        z_clip: float,
        strength: float,
    ) -> None:
        if window_steps < 1:
            raise ValueError("window_steps must be >= 1")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if log_eps <= 0 or std_floor <= 0 or z_clip <= 0:
            raise ValueError("log_eps, std_floor, and z_clip must be positive")
        if strength < 0:
            raise ValueError("strength must be non-negative")
        if strength * z_clip > 1.0:
            raise ValueError(
                "strength * z_clip must be <= 1 so weights cannot reverse "
                "the GRPO advantage direction"
            )
        self.window_steps = int(window_steps)
        self.warmup_steps = int(warmup_steps)
        self.log_eps = float(log_eps)
        self.std_floor = float(std_floor)
        self.z_clip = float(z_clip)
        self.strength = float(strength)
        self._steps: deque[DVACStepStats] = deque(maxlen=self.window_steps)

    def history_summary(self) -> dict[str, float | int]:
        combined = DVACStepStats(
            runner_step=-1,
            count=sum(item.count for item in self._steps),
            value_sum=sum(item.value_sum for item in self._steps),
            value_sq_sum=sum(item.value_sq_sum for item in self._steps),
        )
        return {
            "history_steps": len(self._steps),
            "history_count": combined.count,
            "history_mean": combined.mean,
            "history_std": combined.std,
        }

    def compute_weights(
        self,
        variance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, bool, dict[str, float | int]]:
        if not torch.isfinite(variance).all() or (variance < 0).any():
            raise ValueError("DVAC variance must be finite and non-negative")
        history = self.history_summary()
        warmup = (
            len(self._steps) < self.warmup_steps
            or int(history["history_count"]) == 0
        )
        if warmup or self.strength == 0:
            return (
                torch.ones_like(variance, dtype=torch.float32),
                torch.zeros_like(variance, dtype=torch.float32),
                warmup,
                history,
            )
        log_v = torch.log(variance.float() + self.log_eps)
        denominator = max(float(history["history_std"]), self.std_floor)
        clipped_z = torch.clamp(
            (log_v - float(history["history_mean"])) / denominator,
            -self.z_clip,
            self.z_clip,
        )
        weights = 1.0 + self.strength * clipped_z
        if not torch.isfinite(weights).all():
            raise ValueError("Computed DVAC weights contain NaN or Inf")
        return weights.float(), clipped_z.float(), False, history

    def push(self, stats: DVACStepStats) -> None:
        if stats.count > 0:
            self._steps.append(stats)

    def state_dict(self) -> dict[str, Any]:
        return {
            "window_steps": self.window_steps,
            "warmup_steps": self.warmup_steps,
            "log_eps": self.log_eps,
            "std_floor": self.std_floor,
            "z_clip": self.z_clip,
            "strength": self.strength,
            "steps": [asdict(item) for item in self._steps],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "window_steps": self.window_steps,
            "warmup_steps": self.warmup_steps,
            "log_eps": self.log_eps,
            "std_floor": self.std_floor,
            "z_clip": self.z_clip,
            "strength": self.strength,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(
                    f"DVAC resume config mismatch for {key}: "
                    f"checkpoint={state.get(key)!r}, current={value!r}"
                )
        raw_steps = state.get("steps")
        if not isinstance(raw_steps, list) or len(raw_steps) > self.window_steps:
            raise ValueError("Invalid DVAC resume step history")
        restored: deque[DVACStepStats] = deque(maxlen=self.window_steps)
        for raw in raw_steps:
            item = DVACStepStats(
                runner_step=int(raw["runner_step"]),
                count=int(raw["count"]),
                value_sum=float(raw["value_sum"]),
                value_sq_sum=float(raw["value_sq_sum"]),
            )
            if item.count < 0 or not math.isfinite(item.value_sum) or not math.isfinite(
                item.value_sq_sum
            ):
                raise ValueError("Invalid DVAC resume statistics")
            restored.append(item)
        self._steps = restored
