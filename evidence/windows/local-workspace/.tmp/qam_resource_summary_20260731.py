from __future__ import annotations

import csv
import json
import pathlib
import sys


csv_path = pathlib.Path(sys.argv[1])
rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
if not rows:
    raise SystemExit("resource CSV contains no samples")

integer_fields = [
    "unix_time",
    "cgroup_current_bytes",
    "cgroup_anon_bytes",
    "cgroup_file_bytes",
    "cgroup_oom_events",
    "cgroup_oom_kill_events",
    "disk_available_bytes",
    "gpu0_used_mib",
    "gpu0_util_pct",
    "gpu1_used_mib",
    "gpu1_util_pct",
    "driver_rss_kib",
    "qam_process_rss_kib",
    "compute_process_count",
]
parsed = [{field: int(row[field]) for field in integer_fields} for row in rows]

summary = {
    "sample_count": len(parsed),
    "first_unix_time": parsed[0]["unix_time"],
    "last_unix_time": parsed[-1]["unix_time"],
    "coverage_seconds": parsed[-1]["unix_time"] - parsed[0]["unix_time"],
    "max_cgroup_current_bytes": max(row["cgroup_current_bytes"] for row in parsed),
    "max_cgroup_anon_bytes": max(row["cgroup_anon_bytes"] for row in parsed),
    "max_cgroup_file_bytes": max(row["cgroup_file_bytes"] for row in parsed),
    "min_disk_available_bytes": min(row["disk_available_bytes"] for row in parsed),
    "max_gpu0_used_mib": max(row["gpu0_used_mib"] for row in parsed),
    "max_gpu1_used_mib": max(row["gpu1_used_mib"] for row in parsed),
    "max_gpu0_util_pct": max(row["gpu0_util_pct"] for row in parsed),
    "max_gpu1_util_pct": max(row["gpu1_util_pct"] for row in parsed),
    "max_driver_rss_kib": max(row["driver_rss_kib"] for row in parsed),
    "max_qam_process_rss_kib": max(row["qam_process_rss_kib"] for row in parsed),
    "max_compute_process_count": max(
        row["compute_process_count"] for row in parsed
    ),
    "oom_event_delta": (
        parsed[-1]["cgroup_oom_events"] - parsed[0]["cgroup_oom_events"]
    ),
    "oom_kill_event_delta": (
        parsed[-1]["cgroup_oom_kill_events"]
        - parsed[0]["cgroup_oom_kill_events"]
    ),
    "scope_note": (
        "Recovered monitor began after initialization and covers only the "
        "rollout/update/checkpoint/teardown tail."
    ),
}
print(json.dumps(summary, indent=2, sort_keys=True))
