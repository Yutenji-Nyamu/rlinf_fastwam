from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESOURCE_CSV = ROOT / "runtime" / "resources.csv"
PROCESS_TSV = ROOT / "runtime" / "process_rss.tsv"
JSON_OUT = HERE / "RESOURCE_SUMMARY.json"
PNG_OUT = HERE / "FORMAL_RESOURCE_OVERVIEW.png"

GIB = 1024**3
MIB_PER_GIB = 1024
CGROUP_LIMIT_GIB = 240.0
RECENT_SECONDS = 30 * 60


def fnum(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def timestamp_text(value: object) -> str:
    return pd.Timestamp(value).isoformat()


def linear_slope_per_hour(frame: pd.DataFrame, column: str) -> float | None:
    if len(frame) < 2 or frame["elapsed_s"].nunique() < 2:
        return None
    x = frame["elapsed_s"].to_numpy(dtype=float) / 3600.0
    y = frame[column].to_numpy(dtype=float)
    return fnum(np.polyfit(x, y, 1)[0])


def series_stats(frame: pd.DataFrame, column: str) -> dict[str, object]:
    idx = frame[column].idxmax()
    return {
        "peak": fnum(frame.loc[idx, column]),
        "peak_timestamp": timestamp_text(frame.loc[idx, "timestamp"]),
        "median": fnum(frame[column].median()),
        "p90": fnum(frame[column].quantile(0.90)),
        "latest": fnum(frame.iloc[-1][column]),
    }


def classify_process(row: pd.Series) -> str:
    args = str(row.get("args", ""))
    comm = str(row.get("comm", ""))
    if "EnvWorker" in args:
        return "EnvWorker"
    if "EmbodiedFSDPActor" in args:
        return "FSDPActor"
    if "MultiStepRolloutWorker" in args:
        return "RolloutWorker"
    if "ChannelWorker" in args:
        return "ChannelWorker"
    if "train_embodied_agent.py" in args:
        return "Driver"
    if "WorkerManager" in args:
        return "WorkerManager"
    if "CollectiveManager" in args:
        return "CollectiveManager"
    if "NodeManager" in args:
        return "NodeManager"
    if "DeviceLockManager" in args:
        return "DeviceLockManager"
    if "PortLockManager" in args:
        return "PortLockManager"
    if comm == "raylet" or "/raylet" in args:
        return "Raylet"
    if "gcs_server" in args:
        return "GCS"
    if "dashboard" in args:
        return "RayDashboard"
    if "log_monitor" in args:
        return "RayLogMonitor"
    if comm == "ray::IDLE" or "ray::IDLE" in args:
        return "IdleWorker"
    if "_RemoteNodeProbe" in args:
        return "RemoteNodeProbe"
    return "Other"


def load_and_summarize() -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    resources = pd.read_csv(RESOURCE_CSV)
    resources["timestamp"] = pd.to_datetime(resources["timestamp"])
    numeric_resource_columns = [c for c in resources.columns if c != "timestamp"]
    resources[numeric_resource_columns] = resources[numeric_resource_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    resources = resources.sort_values(["elapsed_s", "gpu_index"]).reset_index(drop=True)

    latest_elapsed = float(resources["elapsed_s"].max())
    recent_cutoff = latest_elapsed - RECENT_SECONDS

    gpu_summary: dict[str, object] = {}
    for gpu_index, gpu in resources.groupby("gpu_index", sort=True):
        gpu = gpu.sort_values("elapsed_s").copy()
        recent = gpu[gpu["elapsed_s"] >= recent_cutoff]
        gpu["gpu_memory_used_gib"] = gpu["gpu_memory_used_mib"] / MIB_PER_GIB
        gpu["gpu_util_5m_mean_pct"] = gpu["gpu_util_pct"].rolling(
            window=150, min_periods=1, center=True
        ).mean()
        recent = recent.copy()
        recent["gpu_memory_used_gib"] = recent["gpu_memory_used_mib"] / MIB_PER_GIB
        recent["gpu_util_5m_mean_pct"] = recent["gpu_util_pct"].rolling(
            window=150, min_periods=1, center=True
        ).mean()
        gpu_summary[str(int(gpu_index))] = {
            "memory_gib": series_stats(gpu, "gpu_memory_used_gib"),
            "utilization_pct": series_stats(gpu, "gpu_util_pct"),
            "recent_30m": {
                "memory_median_gib": fnum(recent["gpu_memory_used_gib"].median()),
                "memory_peak_gib": fnum(recent["gpu_memory_used_gib"].max()),
                "memory_latest_gib": fnum(recent.iloc[-1]["gpu_memory_used_gib"]),
                "util_median_pct": fnum(recent["gpu_util_pct"].median()),
                "util_mean_pct": fnum(recent["gpu_util_pct"].mean()),
                "util_p90_pct": fnum(recent["gpu_util_pct"].quantile(0.90)),
                "util_latest_pct": fnum(recent.iloc[-1]["gpu_util_pct"]),
                "util_5m_rolling_median_pct": fnum(
                    recent["gpu_util_5m_mean_pct"].median()
                ),
                "util_5m_rolling_latest_pct": fnum(
                    recent.iloc[-1]["gpu_util_5m_mean_pct"]
                ),
                "power_median_w": fnum(recent["power_w"].median()),
                "power_peak_w": fnum(recent["power_w"].max()),
                "temperature_peak_c": fnum(recent["temperature_c"].max()),
            },
            "power_w": series_stats(gpu, "power_w"),
            "temperature_c": series_stats(gpu, "temperature_c"),
        }

    cgroup = (
        resources.sort_values(["elapsed_s", "gpu_index"])
        .drop_duplicates(subset=["elapsed_s"], keep="last")
        .copy()
    )
    byte_columns = {
        "cgroup_current_bytes": "current_gib",
        "cgroup_anon_bytes": "anon_gib",
        "cgroup_file_bytes": "file_gib",
    }
    for source, dest in byte_columns.items():
        cgroup[dest] = cgroup[source] / GIB
    recent_cgroup = cgroup[cgroup["elapsed_s"] >= recent_cutoff].copy()
    post_warmup = cgroup[cgroup["elapsed_s"] >= min(3600, latest_elapsed)].copy()

    cgroup_metrics: dict[str, object] = {}
    for column in ["current_gib", "anon_gib", "file_gib"]:
        recent_delta = float(recent_cgroup.iloc[-1][column] - recent_cgroup.iloc[0][column])
        stats = series_stats(cgroup, column)
        stats["recent_30m"] = {
            "start": fnum(recent_cgroup.iloc[0][column]),
            "median": fnum(recent_cgroup[column].median()),
            "peak": fnum(recent_cgroup[column].max()),
            "latest": fnum(recent_cgroup.iloc[-1][column]),
            "delta": fnum(recent_delta),
            "linear_slope_gib_per_hour": linear_slope_per_hour(recent_cgroup, column),
        }
        stats["post_warmup_linear_slope_gib_per_hour"] = linear_slope_per_hour(
            post_warmup, column
        )
        cgroup_metrics[column] = stats

    peak_current = float(cgroup["current_gib"].max())
    latest_current = float(cgroup.iloc[-1]["current_gib"])
    event_columns = ["event_low", "event_high", "event_max", "event_oom", "event_oom_kill"]
    events = {
        column: {
            "initial": int(cgroup.iloc[0][column]),
            "maximum": int(cgroup[column].max()),
            "latest": int(cgroup.iloc[-1][column]),
        }
        for column in event_columns
    }

    availability = {}
    availability_columns = {
        "host_mem_available_kib": "host_memory_available_gib",
        "shm_available_kib": "shm_available_gib",
        "disk_available_kib": "disk_available_gib",
    }
    for source, name in availability_columns.items():
        values = cgroup[source] / (1024**2)
        availability[name] = {
            "initial": fnum(values.iloc[0]),
            "minimum": fnum(values.min()),
            "latest": fnum(values.iloc[-1]),
            "delta_latest_minus_initial": fnum(values.iloc[-1] - values.iloc[0]),
        }

    process = pd.read_csv(PROCESS_TSV, sep="\t")
    process["timestamp"] = pd.to_datetime(process["timestamp"])
    for column in ["pid", "ppid", "rss_kib", "pcpu"]:
        process[column] = pd.to_numeric(process[column], errors="coerce")
    process = process.dropna(subset=["pid", "rss_kib", "timestamp"]).copy()
    process["pid"] = process["pid"].astype(int)
    process["rss_gib"] = process["rss_kib"] / (1024**2)
    process["role"] = process.apply(classify_process, axis=1)
    latest_process_timestamp = process["timestamp"].max()
    process_recent_cutoff = latest_process_timestamp - pd.Timedelta(seconds=RECENT_SECONDS)

    per_pid_rows: list[dict[str, object]] = []
    for pid, frame in process.groupby("pid"):
        frame = frame.sort_values("timestamp")
        peak_idx = frame["rss_gib"].idxmax()
        role = str(frame.iloc[-1]["role"])
        recent_frame = frame[frame["timestamp"] >= process_recent_cutoff].copy()
        if len(recent_frame):
            recent_frame["elapsed_s"] = (
                recent_frame["timestamp"] - recent_frame["timestamp"].min()
            ).dt.total_seconds()
            recent_metrics: dict[str, object] | None = {
                "start_rss_gib": fnum(recent_frame.iloc[0]["rss_gib"]),
                "peak_rss_gib": fnum(recent_frame["rss_gib"].max()),
                "latest_rss_gib": fnum(recent_frame.iloc[-1]["rss_gib"]),
                "delta_gib": fnum(
                    recent_frame.iloc[-1]["rss_gib"]
                    - recent_frame.iloc[0]["rss_gib"]
                ),
                "linear_slope_gib_per_hour": linear_slope_per_hour(
                    recent_frame, "rss_gib"
                ),
            }
        else:
            recent_metrics = None
        per_pid_rows.append(
            {
                "pid": int(pid),
                "role": role,
                "comm": str(frame.iloc[-1]["comm"]),
                "first_seen": timestamp_text(frame.iloc[0]["timestamp"]),
                "last_seen": timestamp_text(frame.iloc[-1]["timestamp"]),
                "peak_rss_gib": fnum(frame.loc[peak_idx, "rss_gib"]),
                "peak_timestamp": timestamp_text(frame.loc[peak_idx, "timestamp"]),
                "initial_observed_rss_gib": fnum(frame.iloc[0]["rss_gib"]),
                "latest_observed_rss_gib": fnum(frame.iloc[-1]["rss_gib"]),
                "delta_latest_minus_initial_gib": fnum(
                    frame.iloc[-1]["rss_gib"] - frame.iloc[0]["rss_gib"]
                ),
                "recent_30m": recent_metrics,
                "present_at_snapshot_end": bool(frame.iloc[-1]["timestamp"] == latest_process_timestamp),
            }
        )
    per_pid_rows.sort(key=lambda x: float(x["peak_rss_gib"]), reverse=True)

    key_roles = {
        "EnvWorker",
        "FSDPActor",
        "RolloutWorker",
        "ChannelWorker",
        "Driver",
        "WorkerManager",
        "CollectiveManager",
        "NodeManager",
        "DeviceLockManager",
        "PortLockManager",
        "Raylet",
    }
    critical_workers = [row for row in per_pid_rows if row["role"] in key_roles]

    aggregate = (
        process.groupby(["timestamp", "role"], as_index=False)["rss_gib"]
        .sum()
        .sort_values("timestamp")
    )
    class_peak = []
    for role, frame in aggregate.groupby("role"):
        idx = frame["rss_gib"].idxmax()
        class_peak.append(
            {
                "role": str(role),
                "peak_aggregate_rss_gib": fnum(frame.loc[idx, "rss_gib"]),
                "peak_timestamp": timestamp_text(frame.loc[idx, "timestamp"]),
            }
        )
    class_peak.sort(key=lambda x: float(x["peak_aggregate_rss_gib"]), reverse=True)

    hourly = cgroup.copy()
    hourly["elapsed_hour"] = (hourly["elapsed_s"] // 3600).astype(int)
    hourly_summary = []
    for hour, frame in hourly.groupby("elapsed_hour"):
        hourly_summary.append(
            {
                "elapsed_hour_bin": int(hour),
                "current_median_gib": fnum(frame["current_gib"].median()),
                "current_peak_gib": fnum(frame["current_gib"].max()),
                "anon_median_gib": fnum(frame["anon_gib"].median()),
                "file_median_gib": fnum(frame["file_gib"].median()),
            }
        )

    current_stats = cgroup_metrics["current_gib"]
    recent_current_delta = float(current_stats["recent_30m"]["delta"])
    post_warmup_current_slope = float(current_stats["post_warmup_linear_slope_gib_per_hour"])
    linear_headroom_hours = (
        (CGROUP_LIMIT_GIB - latest_current) / post_warmup_current_slope
        if post_warmup_current_slope > 0
        else None
    )
    all_events_zero = all(v["maximum"] == 0 for v in events.values())
    summary = {
        "source": {
            "resources_csv": str(RESOURCE_CSV),
            "process_rss_tsv": str(PROCESS_TSV),
            "resource_rows": int(len(resources)),
            "unique_resource_timestamps": int(cgroup["elapsed_s"].nunique()),
            "process_rows": int(len(process)),
            "unique_process_timestamps": int(process["timestamp"].nunique()),
            "snapshot_start": timestamp_text(resources.iloc[0]["timestamp"]),
            "snapshot_end": timestamp_text(resources.iloc[-1]["timestamp"]),
            "elapsed_seconds": int(latest_elapsed),
            "recent_window_seconds": RECENT_SECONDS,
        },
        "gpu": gpu_summary,
        "cgroup": {
            "limit_gib": CGROUP_LIMIT_GIB,
            "metrics": cgroup_metrics,
            "peak_limit_fraction": fnum(peak_current / CGROUP_LIMIT_GIB),
            "latest_limit_fraction": fnum(latest_current / CGROUP_LIMIT_GIB),
            "headroom_at_peak_gib": fnum(CGROUP_LIMIT_GIB - peak_current),
            "headroom_latest_gib": fnum(CGROUP_LIMIT_GIB - latest_current),
            "events": events,
            "hourly_bins": hourly_summary,
        },
        "availability": availability,
        "runtime_health": {
            "driver_alive_all_samples": bool((cgroup["driver_alive"] == 1).all()),
            "driver_alive_latest": int(cgroup.iloc[-1]["driver_alive"]),
            "gpu_compute_process_count_peak": int(cgroup["gpu_compute_process_count"].max()),
            "gpu_compute_process_count_latest": int(
                cgroup.iloc[-1]["gpu_compute_process_count"]
            ),
            "all_memory_events_zero": all_events_zero,
        },
        "process_rss": {
            "critical_workers_by_peak": critical_workers,
            "top_processes_by_peak": per_pid_rows[:15],
            "role_aggregate_peaks": class_peak,
            "note": "RSS is per-process resident memory and can double-count shared pages; cgroup current is the authoritative container total.",
        },
        "assessment": {
            "gpu_capacity_stable": bool(
                max(float(v["memory_gib"]["peak"]) for v in gpu_summary.values()) < 0.5 * 80
            ),
            "no_cgroup_pressure_events": all_events_zero,
            "current_headroom_gib": fnum(CGROUP_LIMIT_GIB - latest_current),
            "recent_30m_current_delta_gib": fnum(recent_current_delta),
            "post_warmup_current_slope_gib_per_hour": fnum(post_warmup_current_slope),
            "linear_hours_to_limit_if_post_warmup_slope_persisted": (
                fnum(linear_headroom_hours) if linear_headroom_hours is not None else None
            ),
            "resource_state": (
                "healthy_now_but_cgroup_growth_risk"
                if all_events_zero and latest_current < 0.85 * CGROUP_LIMIT_GIB
                else "memory_pressure_or_reduced_headroom"
            ),
            "interpretation": (
                "GPU memory is far below capacity and memory events remain zero. "
                "The run has meaningful cgroup headroom at this snapshot, but anon/EnvWorker RSS growth is material. "
                "The hours-to-limit number is a linear risk indicator, not a completion forecast; refresh later before claiming full-run safety."
            ),
        },
    }
    return summary, resources, cgroup, pd.DataFrame(critical_workers)


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def draw_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=16, fill="#ffffff", outline="#d7dee8", width=2)
    draw.text((x0 + 18, y0 + 14), title, font=get_font(23, True), fill="#142033")


def draw_line_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    series: list[tuple[np.ndarray, np.ndarray, str, str]],
    y_min: float,
    y_max: float,
    y_label: str,
    x_max: float,
    y_ticks: list[float],
    reference_lines: list[tuple[float, str, str]] | None = None,
) -> None:
    x0, y0, x1, y1 = box
    left, right, top, bottom = x0 + 70, x1 - 20, y0 + 60, y1 - 45
    grid = "#e7ecf2"
    axis = "#617086"
    label_font = get_font(16)
    small_font = get_font(14)
    for tick in y_ticks:
        yy = bottom - (tick - y_min) / (y_max - y_min) * (bottom - top)
        draw.line((left, yy, right, yy), fill=grid, width=1)
        text_value = f"{tick:g}"
        tw = draw.textbbox((0, 0), text_value, font=small_font)[2]
        draw.text((left - tw - 10, yy - 8), text_value, font=small_font, fill=axis)
    for hour in np.linspace(0, x_max, 6):
        xx = left + hour / max(x_max, 1e-6) * (right - left)
        draw.line((xx, top, xx, bottom), fill=grid, width=1)
        label = f"{hour:.1f}h"
        tw = draw.textbbox((0, 0), label, font=small_font)[2]
        draw.text((xx - tw / 2, bottom + 10), label, font=small_font, fill=axis)
    draw.line((left, top, left, bottom), fill=axis, width=2)
    draw.line((left, bottom, right, bottom), fill=axis, width=2)
    draw.text((x0 + 15, top - 3), y_label, font=small_font, fill=axis)
    if reference_lines:
        for value, label, color in reference_lines:
            if y_min <= value <= y_max:
                yy = bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
                draw.line((left, yy, right, yy), fill=color, width=2)
                tw = draw.textbbox((0, 0), label, font=small_font)[2]
                draw.rectangle((right - tw - 8, yy - 17, right, yy + 1), fill="#ffffff")
                draw.text((right - tw - 4, yy - 16), label, font=small_font, fill=color)
    legend_x = left
    for _, _, label, color in series:
        draw.line((legend_x, y0 + 44, legend_x + 28, y0 + 44), fill=color, width=4)
        draw.text((legend_x + 36, y0 + 33), label, font=label_font, fill="#26354b")
        legend_x += 36 + draw.textbbox((0, 0), label, font=label_font)[2] + 28
    for xs, ys, _, color in series:
        valid = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[valid], ys[valid]
        if len(xs) < 2:
            continue
        stride = max(1, math.ceil(len(xs) / 1800))
        xs, ys = xs[::stride], ys[::stride]
        points = []
        for xxv, yyv in zip(xs, ys):
            xx = left + xxv / max(x_max, 1e-6) * (right - left)
            yy = bottom - (yyv - y_min) / (y_max - y_min) * (bottom - top)
            yy = max(top, min(bottom, yy))
            points.append((xx, yy))
        if len(points) >= 2:
            draw.line(points, fill=color, width=3, joint="curve")


def render_plot(
    summary: dict[str, object],
    resources: pd.DataFrame,
    cgroup: pd.DataFrame,
    critical_workers: pd.DataFrame,
) -> None:
    width, height = 1800, 1220
    image = Image.new("RGB", (width, height), "#f4f7fb")
    draw = ImageDraw.Draw(image)
    draw.text((48, 30), "DVAC formal run resource snapshot", font=get_font(34, True), fill="#111b2d")
    source = summary["source"]
    cg = summary["cgroup"]
    subtitle = (
        f"{source['snapshot_start']} to {source['snapshot_end']}  |  "
        f"elapsed {source['elapsed_seconds'] / 3600:.2f} h  |  "
        f"cgroup {cg['metrics']['current_gib']['latest']:.1f}/{cg['limit_gib']:.0f} GiB, "
        f"headroom {cg['headroom_latest_gib']:.1f} GiB  |  memory events all zero"
    )
    draw.text((48, 79), subtitle, font=get_font(20), fill="#526176")

    panels = {
        "gpu_mem": (40, 125, 880, 575),
        "gpu_util": (920, 125, 1760, 575),
        "cgroup": (40, 605, 1160, 1138),
        "rss": (1200, 605, 1760, 1138),
    }
    draw_panel(draw, panels["gpu_mem"], "GPU memory")
    draw_panel(draw, panels["gpu_util"], "GPU utilization (5-minute rolling mean)")
    draw_panel(draw, panels["cgroup"], "Container memory")
    draw_panel(draw, panels["rss"], "Key process peak RSS")

    colors = {0: "#2563eb", 1: "#ea580c"}
    gpu_mem_series = []
    gpu_util_series = []
    x_max_h = float(resources["elapsed_s"].max()) / 3600.0
    for gpu_index, frame in resources.groupby("gpu_index", sort=True):
        frame = frame.sort_values("elapsed_s")
        x = frame["elapsed_s"].to_numpy(dtype=float) / 3600.0
        memory = frame["gpu_memory_used_mib"].to_numpy(dtype=float) / MIB_PER_GIB
        util = frame["gpu_util_pct"].rolling(window=150, min_periods=1, center=True).mean().to_numpy()
        color = colors[int(gpu_index)]
        gpu_mem_series.append((x, memory, f"GPU{int(gpu_index)}", color))
        gpu_util_series.append((x, util, f"GPU{int(gpu_index)}", color))
    draw_line_chart(
        draw,
        panels["gpu_mem"],
        gpu_mem_series,
        0,
        80,
        "GiB",
        x_max_h,
        [0, 20, 40, 60, 80],
        [(80, "80 GiB capacity", "#9aa6b5")],
    )
    draw_line_chart(
        draw,
        panels["gpu_util"],
        gpu_util_series,
        0,
        100,
        "%",
        x_max_h,
        [0, 25, 50, 75, 100],
    )

    cg_series = [
        (
            cgroup["elapsed_s"].to_numpy(dtype=float) / 3600.0,
            cgroup["current_gib"].to_numpy(dtype=float),
            "current",
            "#7c3aed",
        ),
        (
            cgroup["elapsed_s"].to_numpy(dtype=float) / 3600.0,
            cgroup["anon_gib"].to_numpy(dtype=float),
            "anon",
            "#0891b2",
        ),
        (
            cgroup["elapsed_s"].to_numpy(dtype=float) / 3600.0,
            cgroup["file_gib"].to_numpy(dtype=float),
            "file",
            "#65a30d",
        ),
    ]
    draw_line_chart(
        draw,
        panels["cgroup"],
        cg_series,
        0,
        240,
        "GiB",
        x_max_h,
        [0, 60, 120, 180, 240],
        [(240, "240 GiB limit", "#b91c1c")],
    )

    rss_box = panels["rss"]
    left, right = rss_box[0] + 170, rss_box[2] - 25
    top, bottom = rss_box[1] + 75, rss_box[3] - 48
    workers = critical_workers.copy()
    workers = workers[workers["role"].isin(["EnvWorker", "FSDPActor", "RolloutWorker", "Driver"])]
    workers = workers.sort_values("peak_rss_gib", ascending=False).head(9)
    max_rss = max(80.0, float(workers["peak_rss_gib"].max()) * 1.08 if len(workers) else 80.0)
    for tick in np.linspace(0, max_rss, 5):
        xx = left + tick / max_rss * (right - left)
        draw.line((xx, top, xx, bottom), fill="#e7ecf2", width=1)
        label = f"{tick:.0f}"
        tw = draw.textbbox((0, 0), label, font=get_font(14))[2]
        draw.text((xx - tw / 2, bottom + 10), label, font=get_font(14), fill="#617086")
    draw.text((right - 6, bottom + 29), "peak RSS (GiB)", anchor="ra", font=get_font(14), fill="#617086")
    role_colors = {
        "EnvWorker": "#7c3aed",
        "FSDPActor": "#2563eb",
        "RolloutWorker": "#0891b2",
        "Driver": "#64748b",
    }
    if len(workers):
        row_h = (bottom - top) / len(workers)
        for index, (_, row) in enumerate(workers.iterrows()):
            y = top + index * row_h + row_h * 0.18
            h = row_h * 0.58
            value = float(row["peak_rss_gib"])
            label = f"{row['role']} {int(row['pid'])}"
            label_box = draw.textbbox((0, 0), label, font=get_font(15))
            draw.text((left - (label_box[2] - label_box[0]) - 10, y + h / 2 - 9), label, font=get_font(15), fill="#324258")
            bar_right = left + value / max_rss * (right - left)
            draw.rounded_rectangle((left, y, bar_right, y + h), radius=7, fill=role_colors.get(str(row["role"]), "#64748b"))
            value_text = f"{value:.1f}"
            tx = min(bar_right + 7, right - draw.textbbox((0, 0), value_text, font=get_font(15, True))[2])
            draw.text((tx, y + h / 2 - 9), value_text, font=get_font(15, True), fill="#233147")

    footer = (
        f"Peak cgroup {cg['metrics']['current_gib']['peak']:.1f} GiB; latest {cg['metrics']['current_gib']['latest']:.1f} GiB; "
        f"peak headroom {cg['headroom_at_peak_gib']:.1f} GiB.  "
        f"Disk available {summary['availability']['disk_available_gib']['initial']:.1f} to "
        f"{summary['availability']['disk_available_gib']['latest']:.1f} GiB.  "
        "RSS is per-process; cgroup current is authoritative."
    )
    draw.text((48, 1172), footer, font=get_font(18), fill="#526176")
    image.save(PNG_OUT, optimize=True)


def main() -> None:
    summary, resources, cgroup, critical_workers = load_and_summarize()
    JSON_OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    render_plot(summary, resources, cgroup, critical_workers)
    print(JSON_OUT)
    print(PNG_OUT)


if __name__ == "__main__":
    main()
