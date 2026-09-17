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

"""Optional round schedules and baseline fallback for two-level DVAC weights."""

import math
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any

import torch


def _mapping(value: Any, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _enabled(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _unit_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return result


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def linear_controls_contract(cfg: dict) -> dict:
    """Validate and extract enabled controls for checkpoint compatibility.

    Args:
        cfg: The DVAC configuration, including its mapping and normalization.

    Returns:
        A new dictionary containing only enabled controls. Linear and exponential
        mappings share the controls. Empty or disabled
        controls return an empty dictionary, preserving legacy contracts.

    Raises:
        ValueError: An enabled control or its applicable DVAC mode is invalid.
    """
    # Keep this public name and the actor sidecar key for legacy resume.
    cfg = _mapping(cfg, "dvac")
    contract = {}
    dropout = _mapping(cfg.get("chunk_dropout", {}), "chunk_dropout")
    if _enabled(dropout.get("enabled", False), "chunk_dropout.enabled"):
        probability = _unit_float(
            dropout.get("probability", 0.1), "chunk_dropout.probability"
        )
        seed = _integer(dropout.get("seed", 42), "chunk_dropout.seed")
        if seed >= 2**63:
            raise ValueError("chunk_dropout.seed must be less than 2**63")
        contract["chunk_dropout"] = {
            "enabled": True,
            "probability": probability,
            "seed": seed,
        }

    schedule = _mapping(cfg.get("alpha_schedule", {}), "alpha_schedule")
    if _enabled(schedule.get("enabled", False), "alpha_schedule.enabled"):
        layers = {}
        for layer in ("local", "chunk"):
            name = f"alpha_schedule.{layer}"
            layer_cfg = _mapping(schedule.get(layer, {}), name)
            if not _enabled(layer_cfg.get("enabled", True), f"{name}.enabled"):
                continue
            _unit_float(cfg.get(f"alpha_{layer}", 1.0), f"alpha_{layer}")
            start = _integer(layer_cfg.get("start_step", 1), f"{name}.start_step", 1)
            end = _integer(layer_cfg.get("end_step", 10), f"{name}.end_step", 1)
            if end <= start:
                raise ValueError(f"{name}.end_step must exceed start_step")
            layers[layer] = {
                "enabled": True,
                "start_step": start,
                "end_step": end,
                "end_alpha": _unit_float(
                    layer_cfg.get("end_alpha", 0.2), f"{name}.end_alpha"
                ),
            }
        if layers:
            contract["alpha_schedule"] = {"enabled": True, **layers}

    if contract and (
        cfg.get("mapping", "linear_centered") not in {"linear_centered", "exp_mean"}
        or cfg.get("normalization") != "two_level_group"
    ):
        raise ValueError(
            "DVAC controls require mapping=linear_centered or exp_mean and "
            "normalization=two_level_group"
        )
    return contract


def effective_linear_alphas(
    alpha_local: float,
    alpha_chunk: float,
    controls: dict,
    *,
    runner_step: int,
) -> tuple[float, float]:
    """Return each layer's alpha at a zero-based runner version.

    Schedule steps are one-based training rounds: version 0 is round 1.
    Each enabled layer holds its original alpha until ``start_step``, then
    interpolates linearly to its target at ``end_step`` and holds that target.
    ``controls`` must be the result of :func:`linear_controls_contract`.
    """
    schedule = controls.get("alpha_schedule")
    if not schedule:
        return alpha_local, alpha_chunk
    step = _integer(runner_step, "runner_step") + 1
    effective = []
    for layer, alpha in (("local", alpha_local), ("chunk", alpha_chunk)):
        layer_cfg = schedule.get(layer)
        if not layer_cfg:
            effective.append(alpha)
            continue
        alpha = _unit_float(alpha, f"alpha_{layer}")
        start, end = layer_cfg["start_step"], layer_cfg["end_step"]
        fraction = min(1.0, max(0.0, (step - start) / (end - start)))
        target = layer_cfg["end_alpha"]
        # Preserve exact endpoint values, including a zero target.
        if fraction == 0.0:
            effective.append(alpha)
        elif fraction == 1.0:
            effective.append(target)
        else:
            effective.append(alpha + fraction * (target - alpha))
    return tuple(effective)


def _round_seed(seed: int, runner_step: int) -> int:
    """Mix config seed and runner version without consuming an RNG stream."""
    mask = (1 << 64) - 1
    value = (seed + 0x9E3779B97F4A7C15 * (runner_step + 1)) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & ((1 << 63) - 1)


def apply_chunk_dropout(
    weights: torch.Tensor,
    eligible_mask: torch.Tensor,
    controls: dict,
    *,
    runner_step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restore a Bernoulli-selected subset of chunks to baseline weight one.

    Args:
        weights: Detached DVAC multipliers of shape [T, B, H].
        eligible_mask: Boolean method eligibility of shape [T, B, 1].
        controls: Validated result of :func:`linear_controls_contract`.
        runner_step: Zero-based training round; fixes the mask across repeated
            actor updates on the same globally ordered rollout batch.

    Returns:
        Detached effective weights and a boolean [T, B, 1] dropout mask.
        One coin controls the entire H dimension of each eligible chunk.
        No renormalization or inverted-dropout scaling is applied.

    Raises:
        ValueError: Weight or eligibility dimensions/types are incompatible.
    """
    if weights.ndim != 3 or not weights.is_floating_point():
        raise ValueError("weights must be a floating-point tensor of shape [T, B, H]")
    if (
        eligible_mask.shape != (*weights.shape[:2], 1)
        or eligible_mask.dtype != torch.bool
    ):
        raise ValueError("eligible_mask must be a boolean tensor of shape [T, B, 1]")
    detached = weights.detach()
    eligible = eligible_mask.to(device=weights.device)
    dropped = torch.zeros_like(eligible)
    dropout = controls.get("chunk_dropout")
    if not dropout or dropout["probability"] == 0.0:
        return detached, dropped
    step = _integer(runner_step, "runner_step")
    probability = dropout["probability"]
    if probability == 1.0:
        dropped = eligible
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_round_seed(dropout["seed"], step))
        selected = (
            torch.rand(
                tuple(eligible.shape),
                generator=generator,
                device="cpu",
                dtype=torch.float32,
            )
            < probability
        )
        dropped = selected.to(device=weights.device) & eligible
    return torch.where(dropped, torch.ones_like(detached), detached).detach(), dropped
