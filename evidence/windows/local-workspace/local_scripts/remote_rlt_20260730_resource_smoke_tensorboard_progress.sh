#!/usr/bin/env bash
set -euo pipefail

run_root=/root/autodl-tmp/experiments/rlt_stage2_resource_smoke_8env3c_20260730_v1
python_bin=/root/autodl-tmp/RLinf/.venv/bin/python

RUN_ROOT="$run_root" "$python_bin" -B - <<'PY'
import os
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

events = sorted(Path(os.environ["RUN_ROOT"]).rglob("events.out.tfevents.*"))
print(f"event_files={len(events)}")
for path in events:
    print(f"event_path={path}")
    print(f"event_bytes={path.stat().st_size}")
    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
    accumulator.Reload()
    for tag in sorted(accumulator.Tags().get("scalars", [])):
        values = accumulator.Scalars(tag)
        if not values:
            continue
        last = values[-1]
        print(
            f"scalar={tag}\tcount={len(values)}\t"
            f"last_step={last.step}\tlast_value={last.value}"
        )
PY
