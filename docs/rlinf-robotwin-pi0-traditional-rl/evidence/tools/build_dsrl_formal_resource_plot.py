#!/usr/bin/env python3
"""Build a mobile-readable resource plot from formal-run two-second samples."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def floats(rows: list[dict[str, str]], key: str, divisor: float = 1.0) -> list[float]:
    return [float(row[key]) / divisor for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--cgroup", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--layout",
        choices=("portrait", "landscape"),
        default="portrait",
        help="Use landscape for readable in-conversation previews.",
    )
    args = parser.parse_args()

    resources = read_rows(args.resources)
    cgroup = read_rows(args.cgroup)
    cgroup_by_time = {row["timestamp"]: row for row in cgroup}
    joined = [(row, cgroup_by_time[row["timestamp"]]) for row in resources if row["timestamp"] in cgroup_by_time]
    if not joined:
        raise SystemExit("no timestamp-aligned rows")
    resources = [row for row, _ in joined]
    cgroup = [row for _, row in joined]
    times = [datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S") for row in resources]

    mib_per_gib = 1024.0
    event_start = float(cgroup[0]["memory_events_max"])
    if args.layout == "landscape":
        fig = plt.figure(figsize=(13.5, 8.0), constrained_layout=True)
        grid = fig.add_gridspec(2, 3)
        axes = [
            fig.add_subplot(grid[0, 0:2]),
            fig.add_subplot(grid[0, 2]),
            fig.add_subplot(grid[1, 0]),
            fig.add_subplot(grid[1, 1]),
            fig.add_subplot(grid[1, 2]),
        ]
    else:
        fig, axes = plt.subplots(
            5,
            1,
            figsize=(7.2, 12.8),
            sharex=True,
            constrained_layout=True,
        )
    fig.suptitle(
        f"DSRL pi0 RoboTwin formal resources\n{times[0]:%Y-%m-%d %H:%M:%S}–{times[-1]:%H:%M:%S} CST",
        fontsize=15,
    )

    axes[0].plot(times, floats(resources, "gpu0_memory_mb", mib_per_gib), label="GPU0")
    axes[0].plot(times, floats(resources, "gpu1_memory_mb", mib_per_gib), label="GPU1")
    axes[0].axhline(80, color="0.45", linestyle="--", linewidth=1, label="80 GiB/card")
    axes[0].set_ylabel("GPU memory\nGiB/card")
    axes[0].legend(ncol=3, fontsize=8, loc="upper left")

    axes[1].plot(times, floats(resources, "gpu0_util_pct"), label="GPU0")
    axes[1].plot(times, floats(resources, "gpu1_util_pct"), label="GPU1")
    axes[1].set_ylim(-3, 103)
    axes[1].set_ylabel("GPU util\n%")
    axes[1].legend(ncol=2, fontsize=8, loc="upper left")

    axes[2].plot(times, floats(cgroup, "memory_current_mb", mib_per_gib), label="total")
    axes[2].plot(times, floats(cgroup, "file_mb", mib_per_gib), label="file cache")
    axes[2].plot(times, floats(cgroup, "anon_mb", mib_per_gib), label="anon")
    axes[2].axhline(240, color="0.45", linestyle="--", linewidth=1, label="240 GiB limit")
    axes[2].set_ylabel("cgroup memory\nGiB")
    axes[2].legend(ncol=4, fontsize=8, loc="lower left")

    axes[3].plot(times, floats(resources, "actor_rss_mb", mib_per_gib), label="actor")
    axes[3].plot(times, floats(resources, "rollout_rss_mb", mib_per_gib), label="rollout")
    axes[3].plot(times, floats(resources, "env_rss_mb", mib_per_gib), label="env")
    axes[3].set_ylabel("Worker RSS\nGiB")
    axes[3].legend(ncol=3, fontsize=8, loc="upper left")

    axes[4].plot(
        times,
        [float(row["memory_events_max"]) - event_start for row in cgroup],
        label="memory.events max delta",
    )
    axes[4].set_ylabel("Reclaim/max\nevents")
    axes[4].set_xlabel("Wall clock (CST)")
    axes[4].legend(fontsize=8, loc="upper left")
    oom = max(float(row["memory_events_oom"]) for row in cgroup)
    oom_kill = max(float(row["memory_events_oom_kill"]) for row in cgroup)
    axes[4].text(
        0.99,
        0.06,
        f"OOM={oom:.0f}  OOM-kill={oom_kill:.0f}",
        transform=axes[4].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )

    duration_hours = (times[-1] - times[0]).total_seconds() / 3600

    def time_locator() -> mdates.DateLocator:
        if duration_hours <= 4:
            return mdates.MinuteLocator(byminute=(0, 30))
        if duration_hours <= 12:
            return mdates.HourLocator(interval=1)
        if duration_hours <= 48:
            return mdates.HourLocator(interval=4)
        return mdates.HourLocator(interval=12)

    for axis in axes:
        axis.grid(True, color="0.88", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9)
        axis.xaxis.set_major_locator(time_locator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    plt.close(fig)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
