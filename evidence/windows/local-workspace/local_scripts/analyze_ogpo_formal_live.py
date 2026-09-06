#!/usr/bin/env python3
"""Summarize the current OGPO formal logs downloaded from the server."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
STEP_RE = re.compile(r"Global Step:\s*([0-9]+)/(\d+)")
PAIR_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_./]*)=([-+0-9.eE]+)")


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - pos) + ordered[high] * (pos - low)


def stats(values: list[float]) -> dict[str, float | None]:
    return {
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "last": values[-1] if values else None,
    }


def parse_metric_tables(path: Path) -> list[dict[str, float]]:
    text = ANSI_RE.sub("", path.read_text(encoding="utf-8", errors="replace"))
    matches = list(STEP_RE.finditer(text))
    records: list[dict[str, float]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.start() : end]
        record: dict[str, float] = {
            "global_step": float(match.group(1)),
            "total": float(match.group(2)),
        }
        for key, value in PAIR_RE.findall(block):
            try:
                record["step_seconds" if key == "step" else key] = float(value)
            except ValueError:
                pass
        records.append(record)
    return records


def parse_tensorboard_dump(path: Path) -> dict:
    raw = path.read_bytes()
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8"
    text = raw.decode(encoding, errors="replace")
    marker = text.find('"generated_from"')
    start = text.rfind("{", 0, marker) if marker >= 0 else -1
    if start < 0:
        raise ValueError(f"no JSON object in {path}")
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    return value


def parse_resources(path: Path) -> tuple[list[dict[str, float]], dict]:
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            converted: dict[str, float] = {}
            for key, value in raw.items():
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number):
                    converted[key] = number
            rows.append(converted)

    keys = [
        "host_available_bytes",
        "cgroup_current_bytes",
        "cgroup_anon_bytes",
        "cgroup_file_bytes",
        "shm_used_bytes",
        "disk_available_bytes",
        "gpu0_used_mib",
        "gpu1_used_mib",
        "gpu0_util_pct",
        "gpu1_util_pct",
        "gpu0_power_w",
        "gpu1_power_w",
        "env_rss_kib",
        "actor_rss_kib",
        "rollout_rss_kib",
        "driver_rss_kib",
        "ray_rss_kib",
        "matched_total_rss_kib",
        "compute_process_count",
    ]
    summary = {
        key: stats([row[key] for row in rows if key in row and row[key] >= 0])
        for key in keys
    }
    times = [row["unix_time"] for row in rows if "unix_time" in row]
    gaps = [b - a for a, b in zip(times, times[1:]) if b >= a]

    allocated_gpu_seconds = [0.0, 0.0]
    util_gpu_seconds = [0.0, 0.0]
    energy_joules = [0.0, 0.0]
    for left, right in zip(rows, rows[1:]):
        if "unix_time" not in left or "unix_time" not in right:
            continue
        delta = right["unix_time"] - left["unix_time"]
        if delta < 0 or delta > 10:
            continue
        for gpu in (0, 1):
            mem = left.get(f"gpu{gpu}_used_mib", -1)
            util = left.get(f"gpu{gpu}_util_pct", 0)
            power = left.get(f"gpu{gpu}_power_w", 0)
            if mem > 0:
                allocated_gpu_seconds[gpu] += delta
            util_gpu_seconds[gpu] += max(0.0, util) / 100.0 * delta
            energy_joules[gpu] += max(0.0, power) * delta

    summary["meta"] = {
        "rows": len(rows),
        "first_unix": times[0] if times else None,
        "last_unix": times[-1] if times else None,
        "duration_seconds": times[-1] - times[0] if len(times) >= 2 else 0,
        "cadence_p50_seconds": percentile(gaps, 0.50),
        "cadence_p95_seconds": percentile(gaps, 0.95),
        "gaps_over_5_seconds": sum(gap > 5 for gap in gaps),
        "max_gap_seconds": max(gaps, default=None),
        "oom_max": max((row.get("cgroup_oom_events", 0) for row in rows), default=None),
        "oom_kill_max": max((row.get("cgroup_oom_kill_events", 0) for row in rows), default=None),
        "allocated_gpu_hours": [value / 3600 for value in allocated_gpu_seconds],
        "utilization_equivalent_gpu_hours": [value / 3600 for value in util_gpu_seconds],
        "energy_kwh": [value / 3_600_000 for value in energy_joules],
    }
    return rows, summary


def downsample(rows: list[dict[str, float]], maximum: int = 220) -> list[dict[str, float]]:
    if len(rows) <= maximum:
        return rows
    indices = {round(i * (len(rows) - 1) / (maximum - 1)) for i in range(maximum)}
    return [rows[index] for index in sorted(indices)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root

    tables = parse_metric_tables(root / "metrics.log")
    tb = parse_tensorboard_dump(
        root.parent / "ogpo_formal_20260808_frm0029_metrics_dump.json"
    )
    resources, resource_summary = parse_resources(root / "resources_1s.csv")

    keep_resource_keys = [
        "unix_time",
        "host_available_bytes",
        "cgroup_current_bytes",
        "gpu0_used_mib",
        "gpu1_used_mib",
        "gpu0_util_pct",
        "gpu1_util_pct",
        "gpu0_power_w",
        "gpu1_power_w",
    ]
    sampled_resources = [
        {key: row[key] for key in keep_resource_keys if key in row}
        for row in downsample(resources)
    ]
    result = {
        "metric_tables": tables,
        "tensorboard": tb,
        "resource_summary": resource_summary,
        "resource_downsample": sampled_resources,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
