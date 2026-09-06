#!/usr/bin/env bash
set -euo pipefail

/root/autodl-tmp/RLinf/.venv/bin/python -B - <<'PY'
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

event_path = Path(
    "/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1/"
    "tensorboard/events.out.tfevents.1785324894."
    "autodl-container-nekaqbwt43-6ce5babb.650261.0"
)
accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
accumulator.Reload()
result = {}
for tag in accumulator.Tags().get("scalars", []):
    events = accumulator.Scalars(tag)
    values = [event.value for event in events]
    result[tag] = {
        "count": len(events),
        "first_step": events[0].step,
        "first": values[0],
        "last_step": events[-1].step,
        "last": values[-1],
        "min": min(values),
        "max": max(values),
    }
print(json.dumps({"event_path": str(event_path), "scalars": result}, sort_keys=True))
PY
