#!/usr/bin/env python3
"""Validate and summarize the OGPO smoke 1-second resource CSV."""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path


COLUMNS = [
    "unix_time",
    "host_available_bytes",
    "cgroup_current_bytes",
    "cgroup_anon_bytes",
    "cgroup_file_bytes",
    "cgroup_oom_events",
    "cgroup_oom_kill_events",
    "shm_used_bytes",
    "disk_available_bytes",
    "gpu0_used_mib",
    "gpu0_total_mib",
    "gpu0_util_pct",
    "gpu0_mem_util_pct",
    "gpu0_power_w",
    "gpu1_used_mib",
    "gpu1_total_mib",
    "gpu1_util_pct",
    "gpu1_mem_util_pct",
    "gpu1_power_w",
    "env_rss_kib",
    "env_cpu_pct",
    "actor_rss_kib",
    "actor_cpu_pct",
    "rollout_rss_kib",
    "rollout_cpu_pct",
    "driver_rss_kib",
    "driver_cpu_pct",
    "ray_rss_kib",
    "ray_cpu_pct",
    "matched_total_rss_kib",
    "compute_process_count",
]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: analyze_ogpo_smoke_resources.py resources_1s.csv")
    path = Path(sys.argv[1])
    with path.open(newline="", encoding="utf-8") as handle:
        table = list(csv.reader(handle))
    if not table or table[0] != COLUMNS:
        raise RuntimeError(f"resource CSV header mismatch: {table[:1]}")

    rows: list[dict[str, float]] = []
    for line_number, cells in enumerate(table[1:], 2):
        if len(cells) != len(COLUMNS) or any(cell == "" for cell in cells):
            raise RuntimeError(
                f"invalid resource row {line_number}: {len(cells)} columns"
            )
        rows.append(dict(zip(COLUMNS, map(float, cells), strict=True)))
    if not rows:
        raise RuntimeError("resource CSV contains no samples")
    if any(
        rows[index]["unix_time"] > rows[index + 1]["unix_time"]
        for index in range(len(rows) - 1)
    ):
        raise RuntimeError("resource timestamps are not monotonic")

    def valid(name: str) -> list[tuple[float, float]]:
        return [
            (row["unix_time"], row[name])
            for row in rows
            if row[name] >= 0
        ]

    def extreme(name: str, use_max: bool = True) -> dict[str, float]:
        values = valid(name)
        if not values:
            raise RuntimeError(f"no valid values for {name}")
        timestamp, value = (max if use_max else min)(values, key=lambda item: item[1])
        return {"value": value, "unix_time": timestamp}

    timestamps = [row["unix_time"] for row in rows]
    intervals = [
        right - left
        for left, right in zip(timestamps, timestamps[1:])
        if right > left
    ]
    summary: dict[str, object] = {
        "samples": len(rows),
        "duration_s": timestamps[-1] - timestamps[0],
        "cadence_s_median": statistics.median(intervals) if intervals else 0,
        "cadence_s_max": max(intervals) if intervals else 0,
        "host_available_bytes_min": extreme("host_available_bytes", False),
        "cgroup_current_bytes_peak": extreme("cgroup_current_bytes"),
        "cgroup_anon_bytes_peak": extreme("cgroup_anon_bytes"),
        "cgroup_file_bytes_peak": extreme("cgroup_file_bytes"),
        "shm_used_bytes_peak": extreme("shm_used_bytes"),
        "disk_available_bytes_min": extreme("disk_available_bytes", False),
        "compute_process_count_peak": extreme("compute_process_count"),
        "rss_kib_peak": {
            name: extreme(name)
            for name in (
                "env_rss_kib",
                "actor_rss_kib",
                "rollout_rss_kib",
                "driver_rss_kib",
                "ray_rss_kib",
                "matched_total_rss_kib",
            )
        },
        "gpu": {},
        "oom": {},
    }
    for gpu in (0, 1):
        summary["gpu"][str(gpu)] = {
            suffix: extreme(f"gpu{gpu}_{suffix}")
            for suffix in ("used_mib", "util_pct", "mem_util_pct", "power_w")
        }
    for name in ("cgroup_oom_events", "cgroup_oom_kill_events"):
        values = [value for _, value in valid(name)]
        summary["oom"][name] = {
            "first": values[0],
            "last": values[-1],
            "delta": values[-1] - values[0],
            "monotonic": all(left <= right for left, right in zip(values, values[1:])),
        }

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
