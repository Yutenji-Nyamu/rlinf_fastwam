#!/usr/bin/env bash
set -u

root=/data/chenyiteng/results/rlinf-current-dsrl/formal-dsrl-pi0-robotwin-2gpu4env-200c-20260824-v2/run
python3 - "$root/resource.csv" "$root/dsrl-current-formal-200c-v2/checkpoints" <<'PY'
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

csv_path = Path(sys.argv[1])
checkpoints = Path(sys.argv[2])
with csv_path.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
parsed = [(datetime.fromisoformat(r["timestamp"]), r) for r in rows]
out = []
for ckpt in sorted(checkpoints.glob("global_step_*")):
    mtimes = [p.stat().st_mtime for p in ckpt.rglob("*") if p.is_file()]
    if not mtimes:
        continue
    start = datetime.fromtimestamp(min(mtimes), tz=timezone.utc)
    end = datetime.fromtimestamp(max(mtimes), tz=timezone.utc)
    window = [r for t, r in parsed if (start.timestamp() - 120) <= t.timestamp() <= (end.timestamp() + 120)]
    out.append({
        "checkpoint": ckpt.name,
        "first_file_mtime": start.isoformat(),
        "last_file_mtime": end.isoformat(),
        "window_samples": len(window),
        "gpu6_max_mib": max(float(r["gpu6_used_mib"]) for r in window) if window else None,
        "gpu7_max_mib": max(float(r["gpu7_used_mib"]) for r in window) if window else None,
        "gpu6_util_max": max(float(r["gpu6_util_pct"]) for r in window) if window else None,
        "gpu7_util_max": max(float(r["gpu7_util_pct"]) for r in window) if window else None,
    })
print(json.dumps(out, indent=2))
PY
