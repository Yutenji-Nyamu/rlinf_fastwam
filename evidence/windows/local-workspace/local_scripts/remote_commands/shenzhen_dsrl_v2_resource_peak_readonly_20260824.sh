#!/usr/bin/env bash
set -u

csv=/data/chenyiteng/results/rlinf-current-dsrl/formal-dsrl-pi0-robotwin-2gpu4env-200c-20260824-v2/run/resource.csv
python3 - "$csv" <<'PY'
import csv
import json
import sys
from datetime import datetime

with open(sys.argv[1], newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for r in rows:
    r["peak_mib"] = max(float(r["gpu6_used_mib"]), float(r["gpu7_used_mib"]))
top = sorted(rows, key=lambda r: r["peak_mib"], reverse=True)[:20]
high = [r for r in rows if r["peak_mib"] >= 50000]

clusters = []
for r in high:
    t = datetime.fromisoformat(r["timestamp"])
    if not clusters or (t - clusters[-1][-1][0]).total_seconds() > 30:
        clusters.append([])
    clusters[-1].append((t, r))

out = {
    "top20": [
        {
            "timestamp": r["timestamp"],
            "gpu6_used_mib": float(r["gpu6_used_mib"]),
            "gpu7_used_mib": float(r["gpu7_used_mib"]),
            "gpu6_util_pct": float(r["gpu6_util_pct"]),
            "gpu7_util_pct": float(r["gpu7_util_pct"]),
        }
        for r in top
    ],
    "rows_ge_50000": len(high),
    "clusters_ge_50000": [
        {
            "start": c[0][1]["timestamp"],
            "end": c[-1][1]["timestamp"],
            "samples": len(c),
            "max_mib": max(x[1]["peak_mib"] for x in c),
        }
        for c in clusters
    ],
}
print(json.dumps(out, indent=2))
PY
