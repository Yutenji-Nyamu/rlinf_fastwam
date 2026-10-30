# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""Narrow opt-in budget extension; all model/data/optimizer settings stay fixed."""

import math


def validate_resume(checkpoint, config, windows, world_size, *, extend_completed=False):
    """Keep strict resume by default; extend completed distillation budgets."""
    if checkpoint["windows"] != windows or checkpoint["world_size"] != world_size:
        raise ValueError("Resume data/world-size mismatch")
    old = checkpoint["config"]
    if old == config:
        return None
    if not extend_completed or old["method"] not in {
        "endpoint", "cp", "cp_v2", "onedp", "mfd", "dmd_v2", "mfd_v2"
    }:
        raise ValueError("Resume configuration mismatch")
    changed = {
        key for key in old.keys() | config.keys() if old.get(key) != config.get(key)
    }
    allowed = {"epochs", "checkpoint_steps", "checkpoint_keep_steps"}
    if config.get("retention_mode") == "latest2_best":
        if (
            config.get("checkpoint_keep_last") is not None
            or config.get("checkpoint_keep_steps") != []
        ):
            raise ValueError("Invalid post-evaluation retention configuration")
        allowed |= {"evaluation_steps", "checkpoint_keep_last", "retention_mode"}
    if not changed <= allowed or "epochs" not in changed:
        raise ValueError(f"Extension changed scientific settings: {sorted(changed)}")
    per_epoch = math.ceil(windows / old["batch_size"])
    old_total, new_total = old["epochs"] * per_epoch, config["epochs"] * per_epoch
    if checkpoint["step"] != old_total or new_total <= old_total:
        raise ValueError(
            "Extension requires the final complete checkpoint and a larger budget"
        )
    points = config.get("evaluation_steps", config["checkpoint_keep_steps"])
    if (
        not points
        or points != sorted(set(points))
        or not old_total < points[0] <= points[-1] == new_total
        or config["checkpoint_steps"] != new_total + 1
    ):
        raise ValueError("Invalid extension evaluation/checkpoint schedule")
    if old["method"] in {"onedp", "mfd", "dmd_v2", "mfd_v2"}:
        return {
            "from_step": old_total,
            "to_step": new_total,
            "from_epochs": old["epochs"],
            "to_epochs": config["epochs"],
            "schedule": "keep fixed student and auxiliary learning rates",
            "first_extension_lr": config["lr"],
            "aux_lr": config["aux_lr"],
            "auxiliary_updates_per_student": config.get(
                "auxiliary_updates_per_student", 1
            ),
            "restored": "student, auxiliary, both optimizers, per-rank RNG",
        }
    warm = config["warmup_steps"]
    first_lr = (
        config["lr"]
        * 0.5
        * (1 + math.cos(math.pi * max(0, old_total - warm) / max(1, new_total - warm)))
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
