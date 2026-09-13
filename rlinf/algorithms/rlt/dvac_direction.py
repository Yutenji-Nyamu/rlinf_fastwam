# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Deterministic DVAC direction from the existing RLT critic-update clock."""

from collections.abc import Mapping


def actor_weight_progress(
    update_step: int, warmup_updates: int, ramp_updates: int
) -> tuple[bool, float]:
    """Preserve the original BC/Q course, including its +1 ramp convention."""
    in_warmup = int(update_step) < warmup_updates
    if in_warmup:
        return True, 0.0
    if ramp_updates > 0:
        return False, min(
            1.0,
            max(0.0, float(int(update_step) - warmup_updates + 1) / float(ramp_updates)),
        )
    return False, 1.0


def direction_anchor(direction_cfg: Mapping, actor_cfg: Mapping) -> dict | None:
    """Validate enabled schedules without inserting defaults into old configs."""
    if not direction_cfg or not direction_cfg.get("enable", False):
        return None
    if direction_cfg.get("anchor", "actor_weight_schedule_end") != (
        "actor_weight_schedule_end"
    ):
        raise ValueError("RLT DVAC direction anchor must be actor_weight_schedule_end.")
    if not actor_cfg.get("enable", False):
        raise ValueError("RLT DVAC direction requires an enabled actor_weight_schedule.")
    if direction_cfg.get("scope", "both") not in {"both", "local", "chunk"}:
        raise ValueError("RLT DVAC direction scope must be both, local, or chunk.")
    values = {}
    for name, raw in (
        ("warmup_updates", actor_cfg.get("warmup_updates", 0)),
        ("ramp_updates", actor_cfg.get("ramp_updates", 0)),
        ("transition_updates", direction_cfg.get("transition_updates", 0)),
    ):
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ValueError(f"RLT DVAC direction {name} must be a nonnegative integer.")
        values[name] = raw
    return {
        "actor_weight_schedule": dict(actor_cfg),
        "end_update_step": values["warmup_updates"] + max(values["ramp_updates"] - 1, 0),
    }


def direction_factors(
    update_step: int, direction_cfg: Mapping, actor_cfg: Mapping
) -> tuple[float, float, dict[str, float]]:
    """Return layer signs and diagnostics; disabled schedules preserve old logs."""
    anchor = direction_anchor(direction_cfg, actor_cfg)
    if anchor is None:
        return 1.0, 1.0, {}
    step = int(update_step)
    end = anchor["end_update_step"]
    _, progress = actor_weight_progress(
        step,
        int(actor_cfg.get("warmup_updates", 0)),
        int(actor_cfg.get("ramp_updates", 0)),
    )
    transition = int(direction_cfg.get("transition_updates", 0))
    reached = progress >= 1.0
    if not reached:
        phase = 0.0
    elif transition:
        phase = min(1.0, max(0.0, (step - end) / transition))
    else:
        phase = 1.0
    direction = 1.0 - 2.0 * phase
    scope = direction_cfg.get("scope", "both")
    local = direction if scope in {"both", "local"} else 1.0
    chunk = direction if scope in {"both", "chunk"} else 1.0
    return local, chunk, {
        "rlt_dvac_new/direction_update_step": float(step),
        "rlt_dvac_new/direction_anchor_step": float(end),
        "rlt_dvac_new/direction_local": local,
        "rlt_dvac_new/direction_chunk": chunk,
        "rlt_dvac_new/direction_anchor_reached": float(reached),
    }
