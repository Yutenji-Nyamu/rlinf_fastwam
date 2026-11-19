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

"""Optional DSRL chunk neutralization and round-based strength annealing.

The private dropout seed mixer follows the existing RLT/DVCA controls. Every
sampled replay chunk is eligible, including failures; data is never discarded.
"""

import math
from collections.abc import Mapping
from numbers import Integral, Real

import torch


def _unit(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return value


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _enabled(config, name):
    if not isinstance(config, Mapping) or not isinstance(
        config.get("enabled", False), bool
    ):
        raise ValueError(f"{name} requires a mapping and boolean enabled")
    return config.get("enabled", False)


def dsrl_controls_contract(config: Mapping) -> dict:
    """Canonicalize enabled tricks; disabled controls preserve old checkpoints."""
    _unit(config.get("alpha_chunk", 1.0), "alpha_chunk")
    result = {}
    dropout = config.get("chunk_dropout", {})
    if _enabled(dropout, "chunk_dropout"):
        seed = _integer(dropout.get("seed", 42), "chunk_dropout.seed")
        if seed >= 2**63:
            raise ValueError("chunk_dropout.seed must be less than 2**63")
        result["chunk_dropout"] = {
            "enabled": True,
            "probability": _unit(
                dropout.get("probability", 0.2), "chunk_dropout.probability"
            ),
            "seed": seed,
        }
    schedule = config.get("alpha_schedule", {})
    if _enabled(schedule, "alpha_schedule"):
        start = _integer(schedule.get("start_step", 1), "alpha_schedule.start_step", 1)
        end = _integer(schedule.get("end_step", 200), "alpha_schedule.end_step", 1)
        if end <= start:
            raise ValueError("alpha_schedule.end_step must exceed start_step")
        result["alpha_schedule"] = {
            "enabled": True,
            "start_step": start,
            "end_step": end,
            "end_alpha": _unit(
                schedule.get("end_alpha", 0.0), "alpha_schedule.end_alpha"
            ),
        }
    return result


def _update_seed(seed: int, update_step: int) -> int:
    mask = (1 << 64) - 1
    value = (seed + 0x9E3779B97F4A7C15 * (update_step + 1)) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & ((1 << 63) - 1)


@torch.no_grad()
def apply_dsrl_controls(
    weights: torch.Tensor,
    alpha_chunk: float,
    controls: dict,
    *,
    runner_step: int,
    update_step: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Mix weights toward one, then neutralize selected whole chunks.

    Runner versions are zero based; configured schedule rounds are one based.
    Dropout uses the checkpointed optimizer counter without consuming any global
    RNG, renormalizing weights, rescaling survivors, or filtering replay samples.
    """
    alpha = _unit(alpha_chunk, "alpha_chunk")
    schedule = controls.get("alpha_schedule")
    if schedule:
        step = _integer(runner_step, "runner_step") + 1
        fraction = min(
            1.0,
            max(
                0.0,
                (step - schedule["start_step"])
                / (schedule["end_step"] - schedule["start_step"]),
            ),
        )
        target = schedule["end_alpha"]
        alpha = target if fraction == 1.0 else alpha + fraction * (target - alpha)
    if alpha == 0.0:
        weights = torch.ones_like(weights)
    elif alpha != 1.0:
        weights = 1.0 + alpha * (weights - 1.0)
    dropped = torch.zeros_like(weights, dtype=torch.bool)
    dropout = controls.get("chunk_dropout")
    if dropout and dropout["probability"] > 0:
        step = _integer(update_step, "update_step")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_update_seed(dropout["seed"], step))
        dropped = (
            torch.rand(tuple(weights.shape), generator=generator, device="cpu")
            < dropout["probability"]
        ).to(weights.device)
        weights = torch.where(dropped, torch.ones_like(weights), weights)
    return weights.detach(), {
        "alpha_chunk_effective": alpha,
        "chunk_dropout_fraction": float(dropped.float().mean().item()),
    }
