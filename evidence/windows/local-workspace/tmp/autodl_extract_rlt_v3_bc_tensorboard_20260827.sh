#!/usr/bin/env bash
set -euo pipefail

run=/root/autodl-tmp/experiments/rlt_single_gpu_success_episode_bc_dvac_matched_width_formal480_20260826_v3
event_file=$(find "$run/tensorboard" -maxdepth 1 -type f -name 'events.out.tfevents.*' | head -n 1)
/root/autodl-tmp/RLinf/.venv/bin/python - "$event_file" <<'PY'
import json
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

event_file = sys.argv[1]
acc = EventAccumulator(event_file, size_guidance={"scalars": 0})
acc.Reload()
wanted = [
    tag
    for tag in acc.Tags().get("scalars", [])
    if "rlt_dvac_bc/" in tag
    or tag in {
        "actor/bc_loss",
        "actor/bc_ref_loss",
        "actor/weighted_bc",
        "actor/q_pi",
        "actor/weighted_q",
    }
]
result = {}
for tag in sorted(wanted):
    values = acc.Scalars(tag)
    if not values:
        continue
    latest = values[-1]
    tail = values[-20:]
    result[tag] = {
        "count": len(values),
        "latest_step": latest.step,
        "latest": latest.value,
        "tail20_mean": sum(item.value for item in tail) / len(tail),
    }
print(json.dumps(result, indent=2, sort_keys=True))
PY
