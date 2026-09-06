#!/usr/bin/env bash
set -euo pipefail

run_root=/data/chenyiteng/results/rlinf-current-dsrl/smoke-dsrl-pi0-robotwin-2gpu4env-warm4-20260823
output=$run_root/resource_summary.json

/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -B - \
  "$run_root/fresh/resource.csv" "$run_root/resume/resource.csv" "$output" <<'PY'
import csv
import json
import sys
from pathlib import Path

fresh, resume, output = map(Path, sys.argv[1:])
result = {}
for label, path in (("fresh", fresh), ("resume", resume)):
    rows = list(csv.DictReader(path.open(newline="")))
    live = [row for row in rows if row.get("driver_alive") == "1"]
    assert live
    assert all(int(row.get("cgroup_oom") or 0) == 0 for row in live)
    assert all(int(row.get("cgroup_oom_kill") or 0) == 0 for row in live)
    result[label] = {
        "samples": len(live),
        "min_host_available_gib": min(
            int(row["host_mem_available_kib"]) for row in live
        ) / 1024 / 1024,
        "max_cgroup_gib": max(
            int(row["cgroup_memory_current_bytes"]) for row in live
        ) / 1024**3,
        "max_gpu6_mib": max(int(row.get("gpu6_used_mib") or 0) for row in live),
        "max_gpu7_mib": max(int(row.get("gpu7_used_mib") or 0) for row in live),
        "max_gpu6_util_pct": max(int(row.get("gpu6_util_pct") or 0) for row in live),
        "max_gpu7_util_pct": max(int(row.get("gpu7_util_pct") or 0) for row in live),
    }

output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, sort_keys=True))
PY

printf '%s\n' 'DSRL_CURRENT_SMOKE_RESOURCE_SUMMARY_OK'
