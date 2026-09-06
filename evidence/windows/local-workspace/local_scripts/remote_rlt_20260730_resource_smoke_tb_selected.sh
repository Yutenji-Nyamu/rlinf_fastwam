#!/usr/bin/env bash
set -euo pipefail

run_root=/root/autodl-tmp/experiments/rlt_stage2_resource_smoke_8env3c_20260730_v1
python_bin=/root/autodl-tmp/RLinf/.venv/bin/python

RUN_ROOT="$run_root" "$python_bin" -B - <<'PY'
import os
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

path = next(Path(os.environ["RUN_ROOT"]).rglob("events.out.tfevents.*"))
acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
acc.Reload()
tags = [
    "env/num_trajectories",
    "env/success_once",
    "train/rlt/update_step",
    "train/rlt/critic_updates_run",
    "train/rlt/actor_updates_run",
    "train/sac/actor_loss",
    "train/sac/critic_loss",
    "train/actor/grad_norm",
    "train/critic/grad_norm",
    "time/step",
    "eval/num_trajectories",
    "eval/success_once",
]
for tag in tags:
    if tag not in acc.Tags().get("scalars", []):
        print(f"{tag}\tabsent")
        continue
    values = acc.Scalars(tag)
    print(
        f"{tag}\tcount={len(values)}\t"
        f"steps={[v.step for v in values]}\t"
        f"values={[round(v.value, 8) for v in values]}"
    )
PY
