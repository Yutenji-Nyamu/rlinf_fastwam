#!/usr/bin/env python3
"""Record cgroup memory composition and pressure for one formal RLinf run."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import signal
import time
from datetime import datetime
from pathlib import Path


MIB = 1024 * 1024
GIB = 1024 * MIB
FIELDS = (
    "timestamp",
    "memory_current_mb",
    "memory_limit_mb",
    "anon_mb",
    "file_mb",
    "inactive_file_mb",
    "active_file_mb",
    "shmem_mb",
    "slab_mb",
    "kernel_stack_mb",
    "pagetables_mb",
    "memory_events_max",
    "memory_events_oom",
    "memory_events_oom_kill",
    "memory_psi_some_avg10",
    "memory_psi_full_avg10",
    "autodl_tmp_free_gb",
)


def read_int(path: Path, default: int = 0) -> int:
    try:
        value = path.read_text().strip()
        return default if value == "max" else int(value)
    except (OSError, ValueError):
        return default


def read_pairs(path: Path) -> dict[str, int]:
    try:
        return {
            key: int(value)
            for key, value in (
                line.split(maxsplit=1) for line in path.read_text().splitlines()
            )
        }
    except (OSError, ValueError):
        return {}


def read_pressure(path: Path) -> tuple[float, float]:
    values: dict[str, float] = {}
    try:
        for line in path.read_text().splitlines():
            parts = line.split()
            kind = parts[0]
            metrics = dict(item.split("=", 1) for item in parts[1:])
            values[kind] = float(metrics.get("avg10", 0.0))
    except (OSError, ValueError):
        pass
    return values.get("some", 0.0), values.get("full", 0.0)


def process_start_time(pid: int) -> str | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        tail = stat[stat.rfind(")") + 2 :].split()
        if not tail or tail[0] == "Z":
            return None
        return tail[19]
    except (OSError, IndexError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()
    if args.pid <= 0 or args.interval <= 0:
        parser.error("pid and interval must be positive")

    root = Path("/sys/fs/cgroup")
    expected_start = process_start_time(args.pid)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stop_requested = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    samples = 0
    with args.out.open("w", newline="", buffering=1) as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        while True:
            stat = read_pairs(root / "memory.stat")
            events = read_pairs(root / "memory.events")
            some_avg10, full_avg10 = read_pressure(root / "memory.pressure")
            try:
                free_gb = round(shutil.disk_usage("/root/autodl-tmp").free / GIB, 3)
            except OSError:
                free_gb = 0.0
            limit = read_int(root / "memory.max")
            row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "memory_current_mb": round(read_int(root / "memory.current") / MIB),
                "memory_limit_mb": round(limit / MIB),
                "anon_mb": round(stat.get("anon", 0) / MIB),
                "file_mb": round(stat.get("file", 0) / MIB),
                "inactive_file_mb": round(stat.get("inactive_file", 0) / MIB),
                "active_file_mb": round(stat.get("active_file", 0) / MIB),
                "shmem_mb": round(stat.get("shmem", 0) / MIB),
                "slab_mb": round(stat.get("slab", 0) / MIB),
                "kernel_stack_mb": round(stat.get("kernel_stack", 0) / MIB),
                "pagetables_mb": round(stat.get("pagetables", 0) / MIB),
                "memory_events_max": events.get("max", 0),
                "memory_events_oom": events.get("oom", 0),
                "memory_events_oom_kill": events.get("oom_kill", 0),
                "memory_psi_some_avg10": some_avg10,
                "memory_psi_full_avg10": full_avg10,
                "autodl_tmp_free_gb": free_gb,
            }
            writer.writerow(row)
            output.flush()
            samples += 1

            current_start = process_start_time(args.pid)
            alive = expected_start is not None and current_start == expected_start
            if not alive or stop_requested:
                break
            time.sleep(args.interval)

    print(f"cgroup detail monitor stopped: samples={samples} out={args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
