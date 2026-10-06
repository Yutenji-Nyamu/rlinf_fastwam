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


"""Teacher-signal allocation of the actor's scalar chunk-Q objective."""

import torch

from rlinf.algorithms.rlt.dvac_two_level import build_two_level_success_weights
from rlinf.algorithms.ugrow_signal import UGROW_SIGNAL_SPEC

NORM_SIGNAL_SPEC = {
    "version": 1,
    "kind": "norm_tail5_layers3",
    "steps": 10,
    "tail_steps": 5,
    "tail_layers": 3,
    "readout": "expert_block_output_before_final_norm",
    "reduction": "mean_of_per_token_l2",
    "action_horizon": 50,
}
SIGNAL_KEYS = {
    "ugrow_10_5": "teacher_ugrow_u",
    "norm_tail5_layers3": "teacher_norm",
}


def validate_q_config(config: dict, feature: dict, *, chunk_len: int) -> dict:
    """Freeze the effective method and producer contract before training."""
    result = dict(config)
    source = result.get("signal")
    if source not in SIGNAL_KEYS:
        raise ValueError("Q weighting requires ugrow_10_5 or norm_tail5_layers3.")
    spec = UGROW_SIGNAL_SPEC if source == "ugrow_10_5" else NORM_SIGNAL_SPEC
    if result.get("signal_spec") != spec:
        raise ValueError("Q weighting signal_spec does not match its producer.")
    flags = (
        feature.get("rlt_ugrow_enabled", False),
        feature.get("rlt_norm_enabled", False),
    )
    if flags != ((True, False) if source == "ugrow_10_5" else (False, True)):
        raise ValueError("Enable exactly the selected Q signal producer.")
    if (feature.get("num_steps"), feature.get("action_env_dim"), chunk_len) != (
        10,
        14,
        10,
    ):
        raise ValueError("Q signals require the Clean4 ODE10/D14/C10 contract.")
    if feature.get("rlt_dvac_mode", "off") != "off":
        raise ValueError("Q-only experiments require teacher DVAC collection off.")
    for key, default in (
        ("alpha", 1.0),
        ("temperature", 2.5),
        ("log_eps", 1e-12),
        ("minmax_eps", 1e-6),
    ):
        result.setdefault(key, default)
    # Reuse the reviewed mapper's validation and exact outer reduction.
    chunk_q_weights(torch.zeros(2, chunk_len), result)
    return result


def chunk_q_weights(signal: torch.Tensor, config: dict) -> tuple[torch.Tensor, dict]:
    """Map all sampled rows, including failures, before microbatch splitting."""
    weights, _ = build_two_level_success_weights(
        signal,
        torch.ones(signal.shape[0], dtype=torch.bool, device=signal.device),
        alpha_local=0.0,
        alpha_chunk=float(config["alpha"]),
        log_eps=float(config["log_eps"]),
        minmax_eps=float(config["minmax_eps"]),
        factor_mapping="exp_mean",
        temperature_chunk=float(config["temperature"]),
    )
    weights = weights[:, :1].contiguous().detach()
    ess = weights.sum().square() / (weights.numel() * weights.square().sum())
    metrics = {
        "enabled": 1.0,
        "rows": float(signal.shape[0]),
        "signal_mean": float(signal.mean().item()),
        "signal_std": float(signal.std(unbiased=False).item()),
        "weight_mean": float(weights.mean().item()),
        "weight_std": float(weights.std(unbiased=False).item()),
        "weight_min": float(weights.min().item()),
        "weight_max": float(weights.max().item()),
        "weight_nonunit_fraction": float(
            ((weights - 1).abs() > 1e-6).float().mean().item()
        ),
        "ess_ratio": float(ess.item()),
    }
    return weights, {"rlt_q/" + key: value for key, value in metrics.items()}


def selected_q_signal(obs: dict, config: dict, horizon: int = 10) -> torch.Tensor:
    """Select only the executed C10 from the cached frozen-teacher readout."""
    key = SIGNAL_KEYS[config["signal"]]
    signal = obs.get(key)
    if (
        not isinstance(signal, torch.Tensor)
        or signal.ndim < 2
        or signal.shape[-1] < horizon
    ):
        raise ValueError(f"Missing or invalid cached {key}.")
    if not signal.is_floating_point():
        raise ValueError(f"{key} must be floating point.")
    signal = signal.detach().reshape(-1, signal.shape[-1])[:, :horizon]
    if not torch.isfinite(signal).all() or (signal < 0).any():
        raise ValueError(f"{key} contains nonfinite or negative values.")
    return signal


def weighted_q_mean(q_values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Preserve the scalar-Q gradient and prevent accidental B by B broadcast."""
    if q_values.ndim != 2 or q_values.shape[1] != 1 or weights.shape != q_values.shape:
        raise ValueError("Q and detached chunk weights must both have shape [B,1].")
    if (
        weights.requires_grad
        or not torch.isfinite(weights).all()
        or (weights <= 0).any()
    ):
        raise ValueError("Q weights must be detached, positive and finite.")
    return (weights * q_values).mean()
