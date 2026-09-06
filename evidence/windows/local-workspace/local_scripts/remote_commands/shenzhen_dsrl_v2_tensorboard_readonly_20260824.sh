#!/usr/bin/env bash
set -u

event=/data/chenyiteng/results/rlinf-current-dsrl/formal-dsrl-pi0-robotwin-2gpu4env-200c-20260824-v2/run/tensorboard/events.out.tfevents.1787503673.admin.370987.0
py=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python

"$py" - "$event" <<'PY'
import json
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

event = sys.argv[1]
acc = EventAccumulator(event, size_guidance={"scalars": 0})
acc.Reload()
tags = acc.Tags().get("scalars", [])
out = {}
for tag in sorted(tags):
    vals = acc.Scalars(tag)
    if not vals:
        continue
    out[tag] = {
        "count": len(vals),
        "first_step": vals[0].step,
        "first_value": vals[0].value,
        "last_step": vals[-1].step,
        "last_value": vals[-1].value,
        "min": min(v.value for v in vals),
        "max": max(v.value for v in vals),
    }
print(json.dumps(out, indent=2, sort_keys=True))
PY
