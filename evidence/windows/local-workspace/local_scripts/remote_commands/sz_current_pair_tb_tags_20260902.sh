#!/usr/bin/env bash
set -euo pipefail
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python - <<'PY'
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

runs = {
    "pi05": Path("/data/chenyiteng/results/rlinf-shenzhen/pi05/runs/pi05-grpo-control-formal100-2gpu64x4-g8-b1024-u2-m5-fixed32-eval5-phys45-localshard-v2/tensorboard"),
    "fastwam": Path("/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-formal100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-v1/tensorboard"),
}
for label, root in runs.items():
    events = sorted(root.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    print(f"[{label}] {events[-1]} {events[-1].stat().st_size}")
    ea = EventAccumulator(str(events[-1]), size_guidance={"scalars": 0})
    ea.Reload()
    for tag in ea.Tags().get("scalars", []):
        values = ea.Scalars(tag)
        print(tag, len(values), values[-1].step if values else None, values[-1].value if values else None)
PY
