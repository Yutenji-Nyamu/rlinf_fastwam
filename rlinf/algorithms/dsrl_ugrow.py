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

"""Frozen-behavior solver disagreement and replay-chunk actor weights.

The signal describes a historical latent-conditioned generation. It is not
recomputed for the new latent drawn inside a later SAC update.
"""

import math
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any

import torch


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def make_signal_spec(
    *,
    signal_kind: str = "ugrow_ode10_vs5",
    main_steps: int = 10,
    side_steps: int = 5,
    action_dim: int = 14,
    action_horizon: int = 50,
    chunk_length: int = 10,
    epsilon: float = 1e-8,
) -> dict[str, Any]:
    """Build the strict, JSON-serializable signal identity used by replay."""
    main_steps = _positive_int(main_steps, "main_steps")
    side_steps = _positive_int(side_steps, "side_steps")
    action_dim = _positive_int(action_dim, "action_dim")
    action_horizon = _positive_int(action_horizon, "action_horizon")
    chunk_length = _positive_int(chunk_length, "chunk_length")
    epsilon = _positive_float(epsilon, "epsilon")
    if chunk_length > action_horizon:
        raise ValueError("chunk_length exceeds the predicted action horizon")
    norm_steps = {"norm_residual_t5_l3": (10, 5), "norm_residual_t4_l3": (4, 4)}
    if signal_kind in norm_steps:
        expected_main, tail_steps = norm_steps[signal_kind]
        if main_steps != expected_main:
            raise ValueError("Norm identity does not match its main solver steps")
        return {
            "name": signal_kind,
            "schema_version": 1,
            "main_steps": main_steps,
            "action_dim": action_dim,
            "action_horizon": action_horizon,
            "chunk_length": chunk_length,
            "tail_steps": tail_steps,
            "deep_layers": 3,
            "readout": "post_residual_pre_final_norm",
            "reduction": "mean_of_l2",
            "mask": "submitted_prefix",
            "compute_dtype": "float32",
            "noise": "cast_behavior_latent",
            "extra_solver_steps": 0,
        }
    solver_pairs = {"ugrow_ode10_vs5": (10, 5), "ugrow_ode4_vs2": (4, 2)}
    if signal_kind not in solver_pairs:
        raise ValueError("Unknown DSRL signal identity")
    if (main_steps, side_steps) != solver_pairs[signal_kind]:
        raise ValueError("U identity does not match its complete solver pair")
    return {
        "name": signal_kind,
        "schema_version": 1,
        "main_steps": main_steps,
        "side_steps": side_steps,
        "action_dim": action_dim,
        "action_horizon": action_horizon,
        "chunk_length": chunk_length,
        "epsilon": epsilon,
        "action_space": "model",
        "coordinate_reduction": "mean_population_std_over_rms",
        "mask": "submitted_prefix",
        "compute_dtype": "float32",
        "noise": "same_full_cast_behavior_latent",
    }


U_SPEC = make_signal_spec()


def validate_signal_spec(spec: Mapping) -> dict[str, Any]:
    """Return a canonical copy or reject missing, changed, and unknown fields."""
    if not isinstance(spec, Mapping):
        raise ValueError("DSRL U signal spec must be a mapping")
    if spec.get("name") in {"norm_residual_t5_l3", "norm_residual_t4_l3"}:
        canonical = make_signal_spec(
            signal_kind=spec["name"],
            **{
                key: spec[key]
                for key in (
                    "main_steps",
                    "action_dim",
                    "action_horizon",
                    "chunk_length",
                )
            },
        )
        if dict(spec) != canonical or any(isinstance(v, bool) for v in spec.values()):
            raise ValueError("DSRL Norm signal specification mismatch")
        return canonical
    if set(spec) != set(U_SPEC):
        raise ValueError(
            f"DSRL U spec keys mismatch: {sorted(spec)} != {sorted(U_SPEC)}"
        )
    canonical = make_signal_spec(
        signal_kind=spec.get("name"),
        **{
            key: spec[key]
            for key in (
                "main_steps",
                "side_steps",
                "action_dim",
                "action_horizon",
                "chunk_length",
                "epsilon",
            )
        },
    )
    for key, expected in canonical.items():
        if isinstance(spec[key], bool) or spec[key] != expected:
            raise ValueError(
                f"DSRL U spec mismatch for {key}: {spec[key]!r} != {expected!r}"
            )
    return canonical


@torch.no_grad()
def relative_disagreement(
    actions_10: torch.Tensor,
    actions_5: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-action mean std/RMS from complete same-noise final chunks.

    Inputs are model-space tensors [B,C,D], with padded coordinates excluded by
    the caller. Two zeros are a valid zero score. Nonfinite generation fails
    closed rather than entering replay with an invented uncertainty value.
    """
    epsilon = _positive_float(epsilon, "epsilon")
    if (
        not isinstance(actions_10, torch.Tensor)
        or not isinstance(actions_5, torch.Tensor)
        or actions_10.ndim != 3
        or actions_10.shape != actions_5.shape
        or min(actions_10.shape) <= 0
        or actions_10.device != actions_5.device
        or not actions_10.is_floating_point()
        or not actions_5.is_floating_point()
    ):
        raise ValueError("DSRL U needs matching nonempty floating [B,C,D] tensors")
    a, b = actions_10.detach().float(), actions_5.detach().float()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("DSRL U received nonfinite final model-space actions")
    numerator = (a * 0.5 - b * 0.5).abs()
    denominator = torch.hypot(a / math.sqrt(2.0), b / math.sqrt(2.0))
    score = (numerator / (denominator + epsilon)).mean(dim=-1)
    if not torch.isfinite(score).all():
        raise ValueError("DSRL U generated a nonfinite relative disagreement")
    valid = torch.ones_like(score, dtype=torch.bool)
    return score.detach().contiguous(), valid


@torch.no_grad()
def build_dsrl_u_weights(
    raw_u: torch.Tensor,
    valid: torch.Tensor,
    *,
    temperature: float = 2.5,
    log_eps: float = 1e-12,
    minmax_eps: float = 1e-6,
    alpha_chunk: float = 1.0,
    controls: dict | None = None,
    runner_step: int = 0,
    update_step: int = 0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Map full-global-batch mean(log signal) to detached [B,1] weights.

    The caller must run this once before splitting the global replay batch into
    microbatches. Weights have mean one before optional chunk neutralization.
    False mask entries do not contribute; every row needs at
    least one valid submitted action. Nonfinite or negative valid U is an error.
    """
    temperature = _positive_float(temperature, "temperature")
    log_eps = _positive_float(log_eps, "log_eps")
    minmax_eps = _positive_float(minmax_eps, "minmax_eps")
    if (
        not isinstance(raw_u, torch.Tensor)
        or not isinstance(valid, torch.Tensor)
        or raw_u.ndim != 2
        or min(raw_u.shape) <= 0
        or raw_u.shape != valid.shape
        or raw_u.device != valid.device
        or not raw_u.is_floating_point()
        or valid.dtype != torch.bool
    ):
        raise ValueError("DSRL U weights require floating [B,C] and bool [B,C]")
    # Match the existing RLT/BC exp_mean mapper's stable small-matrix reduction.
    # Signal generation remains FP32; only this [global_B,C] mapping uses FP64.
    values = raw_u.detach().double()
    valid = valid.detach()
    counts = valid.sum(dim=-1)
    if not (counts > 0).all():
        raise ValueError("Every DSRL replay row needs at least one valid U position")
    selected = values[valid]
    if not torch.isfinite(selected).all() or (selected < 0).any():
        raise ValueError("Valid DSRL U must be finite and nonnegative")
    safe_values = torch.where(valid, values, torch.zeros_like(values))
    log_values = torch.log(safe_values + log_eps)
    scores = (
        torch.where(valid, log_values, torch.zeros_like(log_values)).sum(dim=-1)
        / counts
    )
    if not torch.isfinite(scores).all():
        raise ValueError("DSRL U mean-log score is nonfinite")
    span = scores.max() - scores.min()
    if span <= minmax_eps:
        weights = torch.ones_like(scores)
    else:
        unit = (scores - scores.min()) / span
        exponential = torch.exp((unit - unit.max()) / temperature)
        weights = exponential / exponential.mean()
    if not torch.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("DSRL U weights are not finite and strictly positive")
    from rlinf.algorithms.dsrl_chunk_controls import apply_dsrl_controls

    weights, control_metrics = apply_dsrl_controls(
        weights,
        alpha_chunk,
        controls or {},
        runner_step=runner_step,
        update_step=update_step,
    )
    ess = weights.sum().square() / weights.square().sum()
    metrics = {
        **control_metrics,
        "weight_mean": float(weights.mean().item()),
        "weight_min": float(weights.min().item()),
        "weight_max": float(weights.max().item()),
        "weight_std": float(weights.std(unbiased=False).item()),
        "weight_ess": float(ess.item()),
        "weight_ess_fraction": float((ess / weights.numel()).item()),
        "valid_row_count": float(weights.numel()),
        "valid_action_count": float(counts.sum().item()),
        "u_mean": float(selected.mean().item()),
        "u_min": float(selected.min().item()),
        "u_max": float(selected.max().item()),
        "score_span": float(span.item()),
    }
    return weights.unsqueeze(-1).float().detach().contiguous(), metrics
