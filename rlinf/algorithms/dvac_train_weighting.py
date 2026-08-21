"""DVAC-based per-h gradient weighting for embodied policy optimization.

The signal path is intentionally small: rollout workers compute endpoint
variance, actor workers build weights from completed runner-step statistics,
and the straight-through helper changes only the backward contribution of each
future action position.
"""

from __future__ import annotations

import csv
import json
import math
import os
import socket
import subprocess
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


def compute_endpoint_variances(
    z_endpoint: torch.Tensor,
    l_values: Sequence[int] = (2, 3, 4),
) -> dict[int, torch.Tensor]:
    """Return population endpoint variance for each requested denoising tail.

    Args:
        z_endpoint: ``[B, M, H, D_active]`` endpoint previews.
        l_values: Numbers of final denoising previews to aggregate.

    Returns:
        Mapping ``L -> [B, H]`` in float32.
    """

    if z_endpoint.ndim != 4:
        raise ValueError(
            "z_endpoint must have shape [B,M,H,D_active], got "
            f"{tuple(z_endpoint.shape)}"
        )
    z = z_endpoint.detach().float()
    if not torch.isfinite(z).all():
        raise ValueError("z_endpoint contains NaN or Inf")
    num_steps = z.shape[1]
    result: dict[int, torch.Tensor] = {}
    for l_value in l_values:
        l_value = int(l_value)
        if l_value < 2 or l_value > num_steps:
            raise ValueError(f"L={l_value} is invalid for M={num_steps}")
        result[l_value] = z[:, -l_value:].var(
            dim=1, unbiased=False
        ).sum(dim=-1)
    return result


def straight_through_scale_logprobs(
    logprobs: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Keep log-prob values unchanged while scaling per-h gradients."""

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
    loss_mask: torch.Tensor | None,
    *,
    log_eps: float,
) -> torch.Tensor:
    """Compute local ``count,sum,sumsq`` over gradient-participating positions."""

    if variance.ndim != 3:
        raise ValueError(
            f"variance must have shape [T,B,H], got {tuple(variance.shape)}"
        )
    if not torch.isfinite(variance).all() or (variance < 0).any():
        raise ValueError("DVAC variance must be finite and non-negative")
    log_v = torch.log(variance.double() + float(log_eps))
    if loss_mask is None:
        valid = torch.ones_like(log_v, dtype=torch.bool)
    else:
        valid = loss_mask.bool()
        while valid.ndim < log_v.ndim:
            valid = valid.unsqueeze(-1)
        valid = valid.expand_as(log_v)
    selected = log_v[valid]
    if selected.numel() == 0:
        return torch.zeros(3, dtype=torch.float64, device=variance.device)
    return torch.stack(
        (
            torch.tensor(
                float(selected.numel()),
                dtype=torch.float64,
                device=variance.device,
            ),
            selected.sum(),
            selected.square().sum(),
        )
    )


def log_variance_by_h(
    variance: torch.Tensor,
    *,
    log_eps: float,
) -> torch.Tensor:
    """Flatten query axes while retaining the future-action axis ``h``."""

    if variance.ndim != 3:
        raise ValueError(
            f"variance must have shape [T,B,H], got {tuple(variance.shape)}"
        )
    if not torch.isfinite(variance).all() or (variance < 0).any():
        raise ValueError("DVAC variance must be finite and non-negative")
    return torch.log(variance.float() + float(log_eps)).reshape(
        -1, variance.shape[-1]
    )


class DVACRecentStats:
    """Rolling completed-runner-step statistics used by the next step."""

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
                "strength * z_clip must be <= 1 so DVAC weights cannot "
                "reverse the GRPO advantage direction"
            )
        self.window_steps = int(window_steps)
        self.warmup_steps = int(warmup_steps)
        self.log_eps = float(log_eps)
        self.std_floor = float(std_floor)
        self.z_clip = float(z_clip)
        self.strength = float(strength)
        self._steps: deque[DVACStepStats] = deque(maxlen=self.window_steps)

    def history_summary(self) -> dict[str, float | int]:
        count = sum(item.count for item in self._steps)
        value_sum = sum(item.value_sum for item in self._steps)
        value_sq_sum = sum(item.value_sq_sum for item in self._steps)
        combined = DVACStepStats(-1, count, value_sum, value_sq_sum)
        return {
            "history_steps": len(self._steps),
            "history_count": count,
            "history_mean": combined.mean,
            "history_std": combined.std,
        }

    def compute_weights(
        self, variance: torch.Tensor
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
        denom = max(float(history["history_std"]), self.std_floor)
        z = (log_v - float(history["history_mean"])) / denom
        clipped_z = torch.clamp(z, -self.z_clip, self.z_clip)
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
            "history": self.history_summary(),
        }


@dataclass(frozen=True)
class DVACPerHStepValues:
    runner_step: int
    log_v_by_h: torch.Tensor


class DVACPerHResidualStats:
    """Recent per-h median/MAD baseline for R-only gradient weighting."""

    SIGNAL_MODE = "per_h_robust_residual"

    def __init__(
        self,
        *,
        window_steps: int,
        warmup_steps: int,
        log_eps: float,
        scale_floor: float,
        mad_consistency: float,
        residual_clip: float,
        weight_min: float,
        weight_max: float,
    ) -> None:
        if window_steps < 1:
            raise ValueError("window_steps must be >= 1")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if log_eps <= 0 or scale_floor <= 0 or mad_consistency <= 0:
            raise ValueError(
                "log_eps, scale_floor, and mad_consistency must be positive"
            )
        if residual_clip <= 0:
            raise ValueError("residual_clip must be positive")
        if not 0.0 <= weight_min <= 1.0 <= weight_max:
            raise ValueError(
                "Expected 0 <= weight_min <= 1 <= weight_max, got "
                f"[{weight_min}, {weight_max}]"
            )
        self.window_steps = int(window_steps)
        self.warmup_steps = int(warmup_steps)
        self.log_eps = float(log_eps)
        self.scale_floor = float(scale_floor)
        self.mad_consistency = float(mad_consistency)
        self.residual_clip = float(residual_clip)
        self.weight_min = float(weight_min)
        self.weight_max = float(weight_max)
        self._steps: deque[DVACPerHStepValues] = deque(
            maxlen=self.window_steps
        )

    def _history_matrix(self, horizon: int) -> torch.Tensor | None:
        if not self._steps:
            return None
        for item in self._steps:
            if item.log_v_by_h.ndim != 2 or item.log_v_by_h.shape[1] != horizon:
                raise ValueError(
                    "Per-h DVAC history horizon changed: "
                    f"expected H={horizon}, got {tuple(item.log_v_by_h.shape)}"
                )
        return torch.cat(
            [item.log_v_by_h for item in self._steps], dim=0
        ).double()

    def history_summary(self, horizon: int) -> dict[str, Any]:
        matrix = self._history_matrix(horizon)
        if matrix is None:
            missing = torch.full((horizon,), float("nan"), dtype=torch.float32)
            return {
                "history_steps": 0,
                "history_count": 0,
                "history_queries": 0,
                "history_mean": 0.0,
                "history_std": 0.0,
                "history_center_h": missing.clone(),
                "history_mad_h": missing.clone(),
                "history_scale_h": missing.clone(),
            }

        center = torch.quantile(matrix, 0.5, dim=0)
        mad = torch.quantile((matrix - center).abs(), 0.5, dim=0)
        scale = torch.clamp(
            self.mad_consistency * mad,
            min=self.scale_floor,
        )
        return {
            "history_steps": len(self._steps),
            "history_count": int(matrix.numel()),
            "history_queries": int(matrix.shape[0]),
            "history_mean": float(matrix.mean().item()),
            "history_std": float(matrix.std(unbiased=False).item()),
            "history_center_h": center.float(),
            "history_mad_h": mad.float(),
            "history_scale_h": scale.float(),
        }

    def compute_weights(
        self, variance: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, bool, dict[str, Any]]:
        log_v = log_variance_by_h(variance, log_eps=self.log_eps).reshape_as(
            variance
        )
        history = self.history_summary(variance.shape[-1])
        warmup = (
            len(self._steps) < self.warmup_steps
            or int(history["history_count"]) == 0
        )
        if warmup:
            return (
                torch.ones_like(variance, dtype=torch.float32),
                torch.zeros_like(variance, dtype=torch.float32),
                True,
                history,
            )

        center = history["history_center_h"].to(
            device=variance.device, dtype=log_v.dtype
        )
        scale = history["history_scale_h"].to(
            device=variance.device, dtype=log_v.dtype
        )
        residual = (log_v - center) / scale
        clipped_residual = torch.clamp(
            residual, -self.residual_clip, self.residual_clip
        )
        negative_slope = (1.0 - self.weight_min) / self.residual_clip
        positive_slope = (self.weight_max - 1.0) / self.residual_clip
        weights = (
            1.0
            + negative_slope * torch.minimum(
                clipped_residual, torch.zeros_like(clipped_residual)
            )
            + positive_slope * torch.maximum(
                clipped_residual, torch.zeros_like(clipped_residual)
            )
        )
        if not torch.isfinite(weights).all():
            raise ValueError("Computed DVAC residual weights contain NaN or Inf")
        return weights.float(), clipped_residual.float(), False, history

    def push(self, runner_step: int, log_v_by_h: torch.Tensor) -> None:
        values = log_v_by_h.detach().float().cpu().contiguous()
        if values.ndim != 2 or not torch.isfinite(values).all():
            raise ValueError(
                "log_v_by_h must be a finite [N,H] tensor, got "
                f"{tuple(values.shape)}"
            )
        self._steps.append(
            DVACPerHStepValues(
                runner_step=int(runner_step),
                log_v_by_h=values,
            )
        )

    def state_dict(self) -> dict[str, Any]:
        horizon = self._steps[0].log_v_by_h.shape[1] if self._steps else 0
        history = self.history_summary(horizon)
        serializable_history = {
            key: (value.tolist() if torch.is_tensor(value) else value)
            for key, value in history.items()
        }
        return {
            "signal_mode": self.SIGNAL_MODE,
            "window_steps": self.window_steps,
            "warmup_steps": self.warmup_steps,
            "log_eps": self.log_eps,
            "scale_floor": self.scale_floor,
            "mad_consistency": self.mad_consistency,
            "residual_clip": self.residual_clip,
            "weight_min": self.weight_min,
            "weight_max": self.weight_max,
            "steps": [
                {
                    "runner_step": item.runner_step,
                    "queries": int(item.log_v_by_h.shape[0]),
                    "horizon": int(item.log_v_by_h.shape[1]),
                }
                for item in self._steps
            ],
            "history": serializable_history,
        }


def _to_numpy(value: torch.Tensor | None) -> np.ndarray | None:
    if value is None:
        return None
    return value.detach().cpu().contiguous().numpy()


def _git_state(path: str | None) -> dict[str, Any]:
    if not path:
        return {"path": path, "commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", path, "status", "--short", "--", "."],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        return {"path": path, "commit": commit, "dirty": bool(status)}
    except Exception as exc:
        return {
            "path": path,
            "commit": None,
            "dirty": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


class DVACTrainWriter:
    """Rank-local compact artifacts for train-time DVAC analysis."""

    SUMMARY_FIELDS = (
        "runner_step",
        "actor_rank",
        "mode",
        "signal_mode",
        "warmup",
        "history_steps",
        "history_count",
        "history_queries",
        "history_mean",
        "history_std",
        "position_center_median",
        "position_scale_median",
        "current_count",
        "current_mean",
        "current_std",
        "valid_queries_local",
        "weight_min",
        "weight_p05",
        "weight_mean",
        "weight_p50",
        "weight_p95",
        "weight_max",
        "weight_below_one_fraction",
        "weight_above_one_fraction",
        "residual_p05",
        "residual_p50",
        "residual_p95",
        "z_low_clip_fraction",
        "z_high_clip_fraction",
        "positive_adv_weight_mean",
        "negative_adv_weight_mean",
        "actor_grad_norm",
        "actor_clip_fraction",
        "actor_approx_kl",
        "actor_ratio",
        "actor_policy_loss",
    )

    def __init__(
        self,
        output_dir: str,
        *,
        rank: int,
        world_size: int,
        config: Mapping[str, Any],
        repo_path: str | None,
        robotwin_path: str | None,
    ) -> None:
        self.rank = int(rank)
        self.signal_mode = str(config.get("signal_mode", "global_zscore"))
        self.z_clip = float(
            config.get("residual_clip", config.get("z_clip", 2.0))
        )
        self.rank_dir = Path(output_dir) / f"actor_rank{self.rank:02d}"
        if self.rank_dir.exists() and any(self.rank_dir.iterdir()):
            raise FileExistsError(
                f"DVAC train shard directory is not empty: {self.rank_dir}"
            )
        self.rank_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.rank_dir / "runner_step_metrics.csv"
        self.state_path = self.rank_dir / "rolling_stats_state.json"
        manifest = {
            "schema_version": (
                2
                if self.signal_mode == DVACPerHResidualStats.SIGNAL_MODE
                else 1
            ),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "actor_rank": self.rank,
            "actor_world_size": int(world_size),
            "config": dict(config),
            "rlinf_source": _git_state(repo_path),
            "robotwin_source": _git_state(robotwin_path),
        }
        self._write_json_atomic(self.rank_dir / "run_manifest.json", manifest)

    @staticmethod
    def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    @staticmethod
    def _masked_values(
        values: torch.Tensor,
        loss_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if loss_mask is None:
            return values.reshape(-1)
        query_mask = loss_mask.bool()
        while query_mask.ndim > 2:
            query_mask = query_mask.any(dim=-1)
        return values[query_mask].reshape(-1)

    def write_rollout_step(
        self,
        *,
        runner_step: int,
        mode: str,
        warmup: bool,
        variances: Mapping[int, torch.Tensor],
        weights: torch.Tensor,
        clipped_signal: torch.Tensor,
        forward_inputs: Mapping[str, torch.Tensor],
        loss_mask: torch.Tensor | None,
        advantages: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        prev_logprobs: torch.Tensor,
        history: Mapping[str, float | int],
        current_stats: DVACStepStats,
    ) -> dict[str, Any]:
        if dones.shape[0] != weights.shape[0] + 1:
            raise ValueError(
                "Expected one more done boundary than DVAC query steps, got "
                f"dones={tuple(dones.shape)}, weights={tuple(weights.shape)}"
            )
        arrays: dict[str, np.ndarray] = {
            f"v_l{l_value}": _to_numpy(value)
            for l_value, value in variances.items()
        }
        arrays.update(
            {
                "weights": _to_numpy(weights),
                "advantages": _to_numpy(advantages),
                "rewards": _to_numpy(rewards),
                "done_before": _to_numpy(dones[:-1]),
                "done_after": _to_numpy(dones[1:]),
                "loss_mask": (
                    _to_numpy(loss_mask)
                    if loss_mask is not None
                    else np.ones((*weights.shape[:2], 1), dtype=np.bool_)
                ),
                "old_logprob_per_h": _to_numpy(prev_logprobs.sum(dim=-1)),
                "denoise_inds": _to_numpy(forward_inputs["denoise_inds"]),
            }
        )
        if self.signal_mode == DVACPerHResidualStats.SIGNAL_MODE:
            arrays["residual_z"] = _to_numpy(clipped_signal)
            for source_key, output_key in (
                ("history_center_h", "position_center_log_v"),
                ("history_mad_h", "position_mad_log_v"),
                ("history_scale_h", "position_scale_log_v"),
            ):
                value = history.get(source_key)
                if torch.is_tensor(value):
                    arrays[output_key] = _to_numpy(value)
        else:
            arrays["clipped_z"] = _to_numpy(clipped_signal)
        for key, value in forward_inputs.items():
            if key.startswith("dvac_meta_"):
                arrays[key] = _to_numpy(value)
        path = self.rank_dir / f"rollout_step{int(runner_step):04d}.npz"
        if path.exists():
            raise FileExistsError(f"DVAC train rollout shard already exists: {path}")
        temporary = path.with_suffix(".partial.npz")
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)

        valid_weights = self._masked_values(weights, loss_mask)
        valid_z = self._masked_values(clipped_signal, loss_mask)
        query_mask = (
            torch.ones(weights.shape[:2], dtype=torch.bool)
            if loss_mask is None
            else loss_mask.bool().reshape(*loss_mask.shape[:2], -1).any(dim=-1)
        )
        advantage_scalar = advantages.reshape(*advantages.shape[:2], -1).mean(dim=-1)
        per_query_weight = weights.mean(dim=-1)
        positive = query_mask & (advantage_scalar > 0)
        negative = query_mask & (advantage_scalar < 0)

        def mean_or_nan(value: torch.Tensor) -> float:
            return float(value.float().mean().item()) if value.numel() else float("nan")

        if valid_weights.numel():
            quantiles = torch.quantile(
                valid_weights.float(),
                torch.tensor(
                    [0.05, 0.5, 0.95],
                    device=valid_weights.device,
                ),
            )
            weight_min = float(valid_weights.min().item())
            weight_mean = float(valid_weights.float().mean().item())
            weight_max = float(valid_weights.max().item())
            below_one_fraction = float(
                (valid_weights < 1.0).float().mean().item()
            )
            above_one_fraction = float(
                (valid_weights > 1.0).float().mean().item()
            )
            residual_quantiles = torch.quantile(
                valid_z.float(),
                torch.tensor([0.05, 0.5, 0.95], device=valid_z.device),
            )
            low_clip_fraction = float(
                (valid_z <= -self.z_clip).float().mean().item()
            )
            high_clip_fraction = float(
                (valid_z >= self.z_clip).float().mean().item()
            )
        else:
            quantiles = torch.full(
                (3,),
                float("nan"),
                device=weights.device,
            )
            weight_min = weight_mean = weight_max = float("nan")
            below_one_fraction = above_one_fraction = float("nan")
            residual_quantiles = torch.full(
                (3,), float("nan"), device=weights.device
            )
            low_clip_fraction = high_clip_fraction = float("nan")
        center_h = history.get("history_center_h")
        scale_h = history.get("history_scale_h")
        def finite_median_or_nan(value: Any) -> float:
            if not torch.is_tensor(value) or not value.numel():
                return float("nan")
            finite = value.float()[torch.isfinite(value)]
            return (
                float(torch.quantile(finite, 0.5).item())
                if finite.numel()
                else float("nan")
            )

        center_median = finite_median_or_nan(center_h)
        scale_median = finite_median_or_nan(scale_h)
        return {
            "runner_step": int(runner_step),
            "actor_rank": self.rank,
            "mode": mode,
            "signal_mode": self.signal_mode,
            "warmup": int(warmup),
            "history_steps": int(history["history_steps"]),
            "history_count": int(history["history_count"]),
            "history_queries": int(history.get("history_queries", 0)),
            "history_mean": float(history["history_mean"]),
            "history_std": float(history["history_std"]),
            "position_center_median": center_median,
            "position_scale_median": scale_median,
            "current_count": current_stats.count,
            "current_mean": current_stats.mean,
            "current_std": current_stats.std,
            "valid_queries_local": int(query_mask.sum().item()),
            "weight_min": weight_min,
            "weight_p05": float(quantiles[0].item()),
            "weight_mean": weight_mean,
            "weight_p50": float(quantiles[1].item()),
            "weight_p95": float(quantiles[2].item()),
            "weight_max": weight_max,
            "weight_below_one_fraction": below_one_fraction,
            "weight_above_one_fraction": above_one_fraction,
            "residual_p05": float(residual_quantiles[0].item()),
            "residual_p50": float(residual_quantiles[1].item()),
            "residual_p95": float(residual_quantiles[2].item()),
            "z_low_clip_fraction": low_clip_fraction,
            "z_high_clip_fraction": high_clip_fraction,
            "positive_adv_weight_mean": mean_or_nan(per_query_weight[positive]),
            "negative_adv_weight_mean": mean_or_nan(per_query_weight[negative]),
        }

    def write_step_summary(
        self,
        summary: Mapping[str, Any],
        training_metrics: Mapping[str, Any],
        rolling_state: Mapping[str, Any],
    ) -> None:
        row = dict(summary)

        def metric(name: str) -> float:
            value = training_metrics.get(name, float("nan"))
            if torch.is_tensor(value):
                value = value.detach().float().mean().item()
            if isinstance(value, (list, tuple, np.ndarray)):
                value = np.asarray(value, dtype=np.float64).mean()
            return float(value)

        row.update(
            {
                "actor_grad_norm": metric("actor/grad_norm"),
                "actor_clip_fraction": metric("actor/clip_fraction"),
                "actor_approx_kl": metric("actor/approx_kl"),
                "actor_ratio": metric("actor/ratio"),
                "actor_policy_loss": metric("actor/policy_loss"),
            }
        )
        write_header = not self.summary_path.exists()
        with self.summary_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=self.SUMMARY_FIELDS,
                extrasaction="ignore",
            )
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        self._write_json_atomic(self.state_path, rolling_state)
