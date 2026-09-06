#!/usr/bin/env bash
set -u

python=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
stage1_event=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage1-2k-stage2-8env250-20260824-v2/stage1/tensorboard/events.out.tfevents.1787502679.admin.332640.0
stage2_event=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage2-8env250-20260824-v3/tensorboard/events.out.tfevents.1787504104.admin.390911.0
resource=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage2-8env250-20260824-v3/runtime/resource.csv

"$python" - "$stage1_event" "$stage2_event" "$resource" <<'PY'
import csv
import datetime as dt
import json
import statistics
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

for label, event_path in zip(("stage1", "stage2"), sys.argv[1:3]):
    ea = EventAccumulator(event_path, size_guidance={"scalars": 0})
    ea.Reload()
    print(f"--- {label} tensorboard scalars ---")
    for tag in sorted(ea.Tags().get("scalars", [])):
        values = ea.Scalars(tag)
        if not values:
            continue
        last = values[-1]
        print(
            json.dumps(
                {
                    "tag": tag,
                    "count": len(values),
                    "first_step": values[0].step,
                    "first_value": values[0].value,
                    "last_step": last.step,
                    "last_value": last.value,
                    "min": min(item.value for item in values),
                    "max": max(item.value for item in values),
                },
                sort_keys=True,
            )
        )

rows = list(csv.DictReader(Path(sys.argv[3]).open(encoding="utf-8")))
print("--- stage2 resource summary ---")
timestamps = [dt.datetime.fromisoformat(row["timestamp"]) for row in rows]
stall_start = dt.datetime.fromisoformat("2026-08-23T17:39:19+00:00")
print(f"rows={len(rows)} start={timestamps[0].isoformat()} end={timestamps[-1].isoformat()} hours={(timestamps[-1]-timestamps[0]).total_seconds()/3600:.3f}")
for segment, indexes in (
    ("all", range(len(rows))),
    ("before_checkpoint_stall", [i for i, stamp in enumerate(timestamps) if stamp < stall_start]),
    ("during_checkpoint_stall", [i for i, stamp in enumerate(timestamps) if stamp >= stall_start]),
):
    indexes = list(indexes)
    print(f"segment={segment} rows={len(indexes)}")
    for key in (
        "host_mem_available_kib",
        "cgroup_memory_current_bytes",
        "gpu4_used_mib",
        "gpu4_util_pct",
        "gpu5_used_mib",
        "gpu5_util_pct",
    ):
        values = [float(rows[i][key]) for i in indexes]
        values_sorted = sorted(values)
        p95 = values_sorted[min(len(values_sorted) - 1, int(0.95 * (len(values_sorted) - 1)))]
        print(
            f"{key}: min={min(values):.3f} median={statistics.median(values):.3f} "
            f"mean={statistics.fmean(values):.3f} p95={p95:.3f} max={max(values):.3f} last={values[-1]:.3f}"
        )
    print(
        "oom_sum=" + str(sum(int(rows[i]["cgroup_oom"]) for i in indexes))
        + " oom_kill_sum=" + str(sum(int(rows[i]["cgroup_oom_kill"]) for i in indexes))
        + " psi_some_max=" + str(max(float(rows[i]["mem_psi_some_avg10"]) for i in indexes))
        + " psi_full_max=" + str(max(float(rows[i]["mem_psi_full_avg10"]) for i in indexes))
    )
PY

