#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

python=/root/autodl-tmp/RLinf/.venv/bin/python
tb_root=/root/autodl-tmp/experiments/ogpo_robotwin_formal_20260807_v1/tensorboard
runtime_root=/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260807_v1/runtime
export PYTHONDONTWRITEBYTECODE=1

printf 'STATUS_AT\t%s\n' "$(date --iso-8601=seconds)"
printf '%s\n' '=== active worker methods ==='
ps -eo pid=,ppid=,stat=,etimes=,rss=,pcpu=,comm= \
  | awk '$7 ~ /^ray::/ {print}' | sort -k1,1n
printf '%s\n' '=== live tensorboard scalars ==='
"$python" -B - "$tb_root" <<'PY'
import json
import math
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, SCALARS

root = sys.argv[1]
acc = EventAccumulator(root, size_guidance={SCALARS: 0})
acc.Reload()
selected = {}
for tag in sorted(acc.Tags().get("scalars", [])):
    if not (tag.startswith("train/ogpo/") or tag.startswith("eval/")):
        continue
    values = acc.Scalars(tag)
    if not values:
        continue
    item = values[-1]
    selected[tag] = {
        "count": len(values),
        "last_step": item.step,
        "last_value": item.value if math.isfinite(item.value) else str(item.value),
        "last_wall_time": item.wall_time,
    }
print(json.dumps(selected, indent=2, sort_keys=True))
PY
printf '%s\n' '=== monitor tail ==='
tail -n 3 "$runtime_root/resources_1s.csv"
printf '%s\n' '=== cgroup memory events ==='
cat /sys/fs/cgroup/memory.events
