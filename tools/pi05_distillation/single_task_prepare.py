# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""Prepare the native RLinf adjust_bottle smoke; no GPU use or queue mutation."""

import copy
import json
import math
from pathlib import Path

METHODS = ("endpoint", "cp", "onedp", "mfd", "fql_fixed_teacher")


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def training_config(reference, model, manifest, method, micro):
    """Reuse method settings; replace only teacher, single data pool and budget."""
    c = copy.deepcopy(reference)
    c.pop("data", None)
    c.update(
        model=copy.deepcopy(model),
        data_manifest=str(manifest),
        method=method,
        single_task="adjust_bottle",
        batch_size=2 * micro,
        microbatch_per_rank=micro,
        log_every=1,
        checkpoint_steps=100000,
        checkpoint_keep_last=1,
        checkpoint_keep_steps=[],
        smoke_inference_batch=32,
    )
    c["model"]["openpi"]["noise_method"] = "flow_ode"
    c["model"]["openpi"]["distill_target_time"] = method == "cp"
    if method in ("endpoint", "fql_fixed_teacher"):
        c["teacher_steps"] = 5
    if method == "fql_fixed_teacher":
        c.pop("data_manifest")
        c.update(execute_chunk=50, horizon=50, budget_resolved=False)
    elif "warmup_steps" in c:
        windows = read(manifest)["total_windows"]
        c["warmup_steps"] = min(
            500 if method == "cp" else 167,
            max(1, c["epochs"] * math.ceil(windows / c["batch_size"]) // 20),
        )
    return c


def evaluation_config(
    template,
    output,
    seeds_path,
    gpu,
    batch,
    *,
    weights=None,
    method="endpoint",
    collect=False,
    forbidden=(),
):
    c = copy.deepcopy(template)
    c.update(
        suite="pi05_rlinf_single_distill",
        num_envs=batch,
        gpu=gpu,
        output=str(output),
        condition=Path(output).name,
        execute_chunk=50,
        num_steps=5 if collect else 1,
        recording_mode="none",
        record_decisions=False,
        record_full_video=False,
        collect_weak_u=False,
        collect_transitions=collect,
        environment_step_timeout_seconds=1200,
    )
    c["model"]["num_steps"] = c["num_steps"]
    c["model"]["openpi"].update(
        num_steps=c["num_steps"],
        noise_method="flow_ode",
        distill_target_time=bool(weights) and method == "cp",
    )
    c.pop("distill_weights", None)
    if weights:
        c["distill_weights"] = str(weights)
    if collect:
        c.update(global_task_id=0, forbidden_actual_seeds=list(forbidden))
    c["env"].update(total_num_envs=batch, seeds_path=str(seeds_path))
    c["env"]["task_config"]["save_path"] = str(Path(output) / "env")
    c["env"]["video_cfg"].update(
        save_video=False, video_base_dir=str(Path(output) / "unused")
    )
    return c


def selected_micro(method, metrics):
    """One measured larger candidate at most; keep useful headroom for exports."""
    if method == "cp":
        return 80 if metrics["peak_gib"] < 65 else 64
    if metrics["peak_gib"] <= 61:
        return 256
    if metrics["peak_gib"] <= 68:
        return 224
    return 192
