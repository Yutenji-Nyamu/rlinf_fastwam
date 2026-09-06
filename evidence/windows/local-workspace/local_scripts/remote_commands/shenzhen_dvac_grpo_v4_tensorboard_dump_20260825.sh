#!/usr/bin/env bash
set -euo pipefail
RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-grpo-matched-4gpu128x4-b2048-phys4567-v4
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
EVENT=$(find "$RUN/tensorboard" -maxdepth 1 -type f -name 'events.out.tfevents*' | head -n 1)
"$PY" -B - "$EVENT" <<'PY'
import json
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

event = sys.argv[1]
ea = EventAccumulator(event, size_guidance={"scalars": 0})
ea.Reload()
all_tags = sorted(ea.Tags().get("scalars", []))
selected_suffixes = (
    "success_once",
    "approx_kl",
    "clip_fraction",
    "grad_norm",
    "dvac_weight_mean",
    "dvac_weight_sq_mean",
    "dvac_weight_std",
    "dvac_weight_ess_fraction",
    "dvac_z_low_clip_fraction",
    "dvac_z_high_clip_fraction",
    "dvac_warmup",
)
selected = [tag for tag in all_tags if tag.endswith(selected_suffixes)]
payload = {"__all_tags__": all_tags}
for tag in selected:
    payload[tag] = [
        {"step": int(item.step), "wall_time": float(item.wall_time), "value": float(item.value)}
        for item in ea.Scalars(tag)
    ]
print("TENSORBOARD_JSON_BEGIN")
print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
print("TENSORBOARD_JSON_END")
PY
