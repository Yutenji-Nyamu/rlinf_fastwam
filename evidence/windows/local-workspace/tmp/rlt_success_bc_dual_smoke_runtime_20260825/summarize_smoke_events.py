from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def summarize_event(root: Path) -> dict:
    files = sorted(root.glob("events.out.tfevents*"))
    if not files:
        return {"event_file": None, "scalars": {}}
    accumulator = EventAccumulator(str(files[-1]), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalars = {}
    for tag in accumulator.Tags().get("scalars", []):
        events = accumulator.Scalars(tag)
        if events:
            scalars[tag] = {
                "step": events[-1].step,
                "value": events[-1].value,
                "count": len(events),
            }
    return {"event_file": str(files[-1]), "scalars": scalars}


def summarize_resources(path: Path) -> dict:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    numeric = (
        "cgroup_current_bytes",
        "event_high",
        "event_max",
        "event_oom",
        "event_oom_kill",
        "gpu0_used_mib",
        "gpu0_util_pct",
        "gpu1_used_mib",
        "gpu1_util_pct",
        "compute_process_count",
    )
    return {
        "samples": len(rows),
        "max": {key: max(float(row[key]) for row in rows) for key in numeric},
        "first": rows[0],
        "last": rows[-1],
    }


control, method, resources = map(Path, sys.argv[1:4])
summary = {
    "control": summarize_event(control),
    "method": summarize_event(method),
    "resources": summarize_resources(resources),
}
print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
