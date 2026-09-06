from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


monitor_dir = Path(sys.argv[1])
output_json = Path(sys.argv[2])

with (monitor_dir / "resources.csv").open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
assert rows

int_fields = [
    "elapsed_s",
    "gpu_index",
    "gpu_memory_used_mib",
    "gpu_memory_total_mib",
    "gpu_util_pct",
    "gpu_memory_util_pct",
    "temperature_c",
    "cgroup_current_bytes",
    "cgroup_anon_bytes",
    "cgroup_file_bytes",
    "event_low",
    "event_high",
    "event_max",
    "event_oom",
    "event_oom_kill",
    "host_mem_available_kib",
    "shm_available_kib",
    "disk_available_kib",
    "gpu_compute_process_count",
    "driver_alive",
]
for row in rows:
    for key in int_fields:
        row[key] = int(row[key])
    row["power_w"] = float(row["power_w"])

by_gpu = defaultdict(list)
for row in rows:
    by_gpu[row["gpu_index"]].append(row)

gpu_summary = {}
for gpu, gpu_rows in sorted(by_gpu.items()):
    peak_memory = max(gpu_rows, key=lambda row: row["gpu_memory_used_mib"])
    peak_util = max(gpu_rows, key=lambda row: row["gpu_util_pct"])
    gpu_summary[str(gpu)] = {
        "samples": len(gpu_rows),
        "peak_memory_mib": peak_memory["gpu_memory_used_mib"],
        "peak_memory_gib": peak_memory["gpu_memory_used_mib"] / 1024,
        "peak_memory_at": peak_memory["timestamp"],
        "peak_gpu_util_pct": peak_util["gpu_util_pct"],
        "peak_gpu_util_at": peak_util["timestamp"],
        "mean_gpu_util_pct": sum(row["gpu_util_pct"] for row in gpu_rows) / len(gpu_rows),
        "peak_temperature_c": max(row["temperature_c"] for row in gpu_rows),
        "peak_power_w": max(row["power_w"] for row in gpu_rows),
    }

# cgroup/host values are duplicated once per GPU; use GPU 0 rows as time samples.
time_rows = sorted(by_gpu[0], key=lambda row: row["elapsed_s"])
cgroup_peak = max(time_rows, key=lambda row: row["cgroup_current_bytes"])
anon_peak = max(time_rows, key=lambda row: row["cgroup_anon_bytes"])
file_peak = max(time_rows, key=lambda row: row["cgroup_file_bytes"])
event_keys = ["event_low", "event_high", "event_max", "event_oom", "event_oom_kill"]

process_path = monitor_dir / "process_rss.tsv"
process_rows = []
with process_path.open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        if not row or not row.get("pid"):
            continue
        row["pid"] = int(row["pid"])
        row["ppid"] = int(row["ppid"])
        row["rss_kib"] = int(row["rss_kib"])
        row["pcpu"] = float(row["pcpu"])
        process_rows.append(row)

peak_by_pid = {}
for row in process_rows:
    current = peak_by_pid.get(row["pid"])
    if current is None or row["rss_kib"] > current["rss_kib"]:
        peak_by_pid[row["pid"]] = row
top_processes = []
for row in sorted(peak_by_pid.values(), key=lambda row: row["rss_kib"], reverse=True)[:15]:
    top_processes.append(
        {
            "pid": row["pid"],
            "ppid": row["ppid"],
            "peak_rss_kib": row["rss_kib"],
            "peak_rss_gib": row["rss_kib"] / 1024 / 1024,
            "pcpu_at_peak": row["pcpu"],
            "comm": row["comm"],
            "args": row["args"],
            "timestamp": row["timestamp"],
        }
    )

summary = {
    "resource_rows": len(rows),
    "time_samples": len(time_rows),
    "elapsed_first_s": time_rows[0]["elapsed_s"],
    "elapsed_last_s": time_rows[-1]["elapsed_s"],
    "gpu": gpu_summary,
    "cgroup": {
        "initial_current_bytes": time_rows[0]["cgroup_current_bytes"],
        "final_sample_current_bytes": time_rows[-1]["cgroup_current_bytes"],
        "peak_current_bytes": cgroup_peak["cgroup_current_bytes"],
        "peak_current_gib": cgroup_peak["cgroup_current_bytes"] / 1024**3,
        "peak_current_at": cgroup_peak["timestamp"],
        "peak_anon_bytes": anon_peak["cgroup_anon_bytes"],
        "peak_anon_gib": anon_peak["cgroup_anon_bytes"] / 1024**3,
        "peak_file_bytes": file_peak["cgroup_file_bytes"],
        "peak_file_gib": file_peak["cgroup_file_bytes"] / 1024**3,
        "events_first": {key: time_rows[0][key] for key in event_keys},
        "events_last": {key: time_rows[-1][key] for key in event_keys},
    },
    "host": {
        "minimum_mem_available_kib": min(row["host_mem_available_kib"] for row in time_rows),
        "minimum_shm_available_kib": min(row["shm_available_kib"] for row in time_rows),
        "minimum_disk_available_kib": min(row["disk_available_kib"] for row in time_rows),
        "peak_gpu_compute_process_count": max(row["gpu_compute_process_count"] for row in time_rows),
    },
    "top_processes_by_peak_rss": top_processes,
}
output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
print("RESOURCE_SUMMARY_PASS=1")
