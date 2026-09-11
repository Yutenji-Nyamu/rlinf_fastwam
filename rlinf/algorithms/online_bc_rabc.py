# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Detached SARM RA-BC weights for progress measured in remaining seconds.

The caller updates statistics once per newly admitted unique chunk. Sampling and
loss weighting never update them. The caller also owns scorer/cache identity and
must apply the explicit clean-mix/all-zero policy before an optimizer step.
"""

import math
from dataclasses import dataclass
from numbers import Real

import torch


@dataclass(frozen=True)
class RABCConfig:
    kappa_seconds: float
    epsilon_signal: float = 1e-6
    epsilon_weight: float = 1e-6
    clean_mix: float = 0.0

    def __post_init__(self):
        for name in ("kappa_seconds", "epsilon_signal", "epsilon_weight"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"RA-BC {name} must be finite and positive.")
        if (
            isinstance(self.clean_mix, bool)
            or not isinstance(self.clean_mix, Real)
            or not math.isfinite(self.clean_mix)
            or not 0 <= self.clean_mix <= 1
        ):
            raise ValueError("RA-BC clean_mix must be finite and in [0, 1].")

    @property
    def all_zero_policy(self) -> str:
        return "skip" if self.clean_mix == 0 else "clean"


def _vector(values, *, allow_empty=False):
    if isinstance(values, torch.Tensor):
        if values.is_complex() or values.dtype == torch.bool:
            raise ValueError("RA-BC values must be real numbers, not complex/bool.")
        values = values.detach().to(dtype=torch.float64)
    else:
        values = torch.as_tensor(values, dtype=torch.float64)
    if values.ndim != 1 or (not allow_empty and values.numel() == 0):
        raise ValueError("RA-BC requires a nonempty one-dimensional vector.")
    if not torch.isfinite(values).all():
        raise ValueError("RA-BC values must all be finite.")
    return values


class RABCStats:
    """Mergeable float64 Welford moments with a strict seconds/population schema."""

    _IDENTITY = {
        "format": "online_bc_rabc_stats",
        "version": 1,
        "units": "seconds",
        "variance": "population",
    }

    def __init__(self):
        self.count = 0
        self.raw_mean = 0.0
        self.m2 = 0.0

    def update(self, deltas) -> None:
        values = _vector(deltas, allow_empty=True).cpu()
        size = values.numel()
        if size == 0:
            return
        batch_mean = values.mean().item()
        batch_m2 = (values - batch_mean).square().sum().item()
        count = self.count + size
        if self.count == 0:
            mean, m2 = batch_mean, batch_m2
        else:
            shift = batch_mean - self.raw_mean
            mean = self.raw_mean + shift * (size / count)
            m2 = self.m2 + batch_m2 + shift * shift * (self.count * size / count)
        if not math.isfinite(mean) or not math.isfinite(m2):
            raise ValueError("RA-BC moment update overflowed; statistics unchanged.")
        self.count, self.raw_mean, self.m2 = count, mean, m2

    @property
    def population_std(self) -> float:
        return math.sqrt(self.m2 / self.count) if self.count else 0.0

    def mapping_moments(self, config: RABCConfig) -> tuple[float, float]:
        if self.count == 0:
            raise ValueError("RA-BC weights require admitted progress statistics.")
        # Clamp only this readout. The stored mean/M2 must remain raw moments.
        return max(self.raw_mean, 0.0), max(self.population_std, config.epsilon_signal)

    def state_dict(self) -> dict:
        return {
            **self._IDENTITY,
            "count": self.count,
            "raw_mean": self.raw_mean,
            "m2": self.m2,
        }

    def load_state_dict(self, state: dict) -> None:
        expected_keys = set(self._IDENTITY) | {"count", "raw_mean", "m2"}
        if not isinstance(state, dict) or set(state) != expected_keys:
            raise ValueError("Invalid RA-BC statistics schema.")
        for key, expected in self._IDENTITY.items():
            if type(state[key]) is not type(expected) or state[key] != expected:
                raise ValueError(f"Incompatible RA-BC statistics identity: {key}.")
        count, mean, m2 = state["count"], state["raw_mean"], state["m2"]
        if type(count) is not int or count < 0:
            raise ValueError("Invalid RA-BC statistics count.")
        for value in (mean, m2):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError("Invalid RA-BC statistics moments.")
        if m2 < 0 or (count == 0 and (mean != 0 or m2 != 0)) or (count == 1 and m2 != 0):
            raise ValueError("Inconsistent RA-BC statistics moments/count.")
        # Validate fully before changing live state.
        self.count, self.raw_mean, self.m2 = count, float(mean), float(m2)


def raw_weights(deltas, stats: RABCStats, config: RABCConfig) -> torch.Tensor:
    """Return detached float64 weights in [0, 1], preserving a tensor's device."""
    values = _vector(deltas)
    mean, std = stats.mapping_moments(config)
    soft = ((values - (mean - 2 * std)) / (4 * std + config.epsilon_signal)).clamp(0, 1)
    weights = torch.where(values > config.kappa_seconds, torch.ones_like(soft), soft)
    return torch.where(values < 0, torch.zeros_like(weights), weights).detach()


def normalized_batch_weights(raw, config: RABCConfig) -> tuple[torch.Tensor, dict]:
    """Normalize one complete optimizer batch, before splitting microbatches.

    Averaging ``normalized * per_query_loss`` yields sum(w*loss)/(sum(w)+eps).
    Do not normalize each microbatch again. A zero batch is returned as zeros with
    ``skip_update=True``; skipping Adam is the caller's responsibility.
    """
    weights = _vector(raw)
    if ((weights < 0) | (weights > 1)).any():
        raise ValueError("RA-BC raw weights must lie in [0, 1].")
    size = weights.numel()
    total = weights.sum()
    squared_total = weights.square().sum()
    all_zero = bool(total == 0)
    normalized = weights * (size / (total + config.epsilon_weight))
    diagnostics = {
        "batch_size": size,
        "raw_weight_mean": weights.mean().item(),
        "raw_weight_std": weights.std(unbiased=False).item(),
        "raw_weight_sum": total.item(),
        "num_zero_weight": int((weights == 0).sum()),
        "num_full_weight": int((weights == 1).sum()),
        "zero_fraction": (weights == 0).double().mean().item(),
        "full_fraction": (weights == 1).double().mean().item(),
        "effective_sample_size": 0.0 if all_zero else (total.square() / squared_total).item(),
        "normalized_weight_mean": normalized.mean().item(),
        "normalized_weight_sum": normalized.sum().item(),
        "all_zero": all_zero,
        "skip_update": all_zero,
    }
    return normalized.detach(), diagnostics


def mix_normalized_batch_weights(
    normalized: torch.Tensor, config: RABCConfig
) -> tuple[torch.Tensor, dict]:
    """Mix *after* the existing full-batch normalization; never renormalize.

    A valid all-zero batch falls back to full clean BC only when clean_mix > 0.
    Invalid scores/masks must be rejected by their normal validation path, not
    converted to clean data. With c=0 return the original detached tensor.
    """
    if not isinstance(normalized, torch.Tensor):
        raise ValueError("RA-BC normalized weights must be a tensor.")
    weights = _vector(normalized)
    if (weights < 0).any():
        raise ValueError("RA-BC normalized weights must be nonnegative.")
    all_zero = bool(weights.sum() == 0)
    fallback = all_zero and config.clean_mix > 0
    if config.clean_mix == 0:
        final = normalized.detach()
    elif fallback:
        final = torch.ones_like(weights)
    else:
        final = config.clean_mix + (1 - config.clean_mix) * weights
    total, squared_total = final.sum(), final.square().sum()
    diagnostics = {
        "clean_mix": float(config.clean_mix),
        "base_skip_update": all_zero,
        "skip_update": all_zero and not fallback,
        "clean_fallback": fallback,
        "final_weight_mean": final.mean().item(),
        "final_weight_min": final.min().item(),
        "final_weight_max": final.max().item(),
        "final_weight_std": final.std(unbiased=False).item(),
        "final_weight_zero_fraction": (final == 0).double().mean().item(),
        "final_effective_sample_size": (
            0.0 if bool(squared_total == 0) else (total.square() / squared_total).item()
        ),
    }
    return final.detach(), diagnostics
