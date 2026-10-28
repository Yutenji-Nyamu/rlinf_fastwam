# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""Narrow opt-in budget extension; all model/data/optimizer settings stay fixed."""

import math


def validate_resume(checkpoint, config, windows, world_size, *, extend_completed=False):
    """Keep strict resume by default, allowing only a completed CP/direct budget."""
    if checkpoint["windows"] != windows or checkpoint["world_size"] != world_size:
        raise ValueError("Resume data/world-size mismatch")
    old = checkpoint["config"]
    if old == config:
        return None
    if not extend_completed or old["method"] not in {"endpoint", "cp"}:
        raise ValueError("Resume configuration mismatch")
    changed = {key for key in old.keys() | config.keys() if old.get(key) != config.get(key)}
    allowed = {"epochs", "checkpoint_steps", "checkpoint_keep_steps"}
    if not changed <= allowed or "epochs" not in changed:
        raise ValueError(f"Extension changed scientific settings: {sorted(changed)}")
    per_epoch = math.ceil(windows / old["batch_size"])
    old_total, new_total = old["epochs"] * per_epoch, config["epochs"] * per_epoch
    if checkpoint["step"] != old_total or new_total <= old_total:
        raise ValueError("Extension requires the final complete checkpoint and a larger budget")
    points = config["checkpoint_keep_steps"]
    if (
        not points
        or points != sorted(set(points))
        or not old_total < points[0] <= points[-1] == new_total
        or config["checkpoint_steps"] != new_total + 1
    ):
        raise ValueError("Invalid extension evaluation/checkpoint schedule")
    warm = config["warmup_steps"]
    first_lr = config["lr"] * 0.5 * (
        1 + math.cos(math.pi * max(0, old_total - warm) / max(1, new_total - warm))
    )
    return {
        "from_step": old_total,
        "to_step": new_total,
        "from_epochs": old["epochs"],
        "to_epochs": config["epochs"],
        "schedule": "stretch cosine to new total; keep original absolute warmup and global step",
        "first_extension_lr": first_lr,
        "restored": "student, EMA if CP, optimizer, per-rank RNG",
    }
