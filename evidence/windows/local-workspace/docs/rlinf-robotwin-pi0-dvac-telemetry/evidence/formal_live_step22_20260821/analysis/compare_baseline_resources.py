from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from analyze_resources import draw_panel, fnum, get_font


HERE = Path(__file__).resolve().parent
SNAPSHOT_ROOT = HERE.parent
WORKSPACE = Path(__file__).resolve().parents[5]

FORMAL_RESOURCE_CSV = SNAPSHOT_ROOT / "runtime" / "resources.csv"
FORMAL_PROCESS_TSV = SNAPSHOT_ROOT / "runtime" / "process_rss.tsv"
BASELINE_ROOT = WORKSPACE / "audits" / "20260717-084926-grpo-current"
BASELINE_RESOURCE_CSV = BASELINE_ROOT / "resources.csv"
BASELINE_ANALYSIS_JSON = BASELINE_ROOT / "analysis.json"

CSV_OUT = HERE / "FORMAL_VS_BASELINE_RESOURCE_10MIN.csv"
JSON_OUT = HERE / "FORMAL_VS_BASELINE_RESOURCE_SUMMARY.json"
PNG_OUT = HERE / "FORMAL_VS_BASELINE_RESOURCE.png"

GIB = 1024**3
MIB_PER_GIB = 1024
FIRST_WINDOW_HOURS = 10.0
BIN_SECONDS = 10 * 60


def stat_block(frame: pd.DataFrame, column: str) -> dict[str, float | str]:
    valid = frame.dropna(subset=[column]).sort_values("elapsed_s")
    idx = valid[column].idxmax()
    return {
        "start": fnum(valid.iloc[0][column]),
        "median": fnum(valid[column].median()),
        "peak": fnum(valid.loc[idx, column]),
        "peak_elapsed_h": fnum(valid.loc[idx, "elapsed_s"] / 3600.0),
        "end": fnum(valid.iloc[-1][column]),
        "peak_growth_from_start": fnum(valid.loc[idx, column] - valid.iloc[0][column]),
        "growth_end_minus_start": fnum(valid.iloc[-1][column] - valid.iloc[0][column]),
    }


def load_formal() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(FORMAL_RESOURCE_CSV)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    numeric = [column for column in raw.columns if column != "timestamp"]
    raw[numeric] = raw[numeric].apply(pd.to_numeric, errors="coerce")
    raw = raw.sort_values(["elapsed_s", "gpu_index"])

    cgroup = raw.drop_duplicates("elapsed_s", keep="last").copy()
    cgroup["cgroup_current_gib"] = cgroup["cgroup_current_bytes"] / GIB

    gpu = raw.pivot_table(
        index=["timestamp", "elapsed_s"],
        columns="gpu_index",
        values="gpu_memory_used_mib",
        aggfunc="last",
    ).reset_index()
    gpu.columns = [
        "timestamp",
        "elapsed_s",
        "gpu0_memory_mib",
        "gpu1_memory_mib",
    ]
    gpu["gpu0_memory_gib"] = gpu["gpu0_memory_mib"] / MIB_PER_GIB
    gpu["gpu1_memory_gib"] = gpu["gpu1_memory_mib"] / MIB_PER_GIB

    formal = cgroup[["timestamp", "elapsed_s", "cgroup_current_gib"]].merge(
        gpu[["elapsed_s", "gpu0_memory_gib", "gpu1_memory_gib"]],
        on="elapsed_s",
        how="left",
    )
    formal["elapsed_s"] = formal["elapsed_s"].astype(float)

    process = pd.read_csv(FORMAL_PROCESS_TSV, sep="\t")
    process["timestamp"] = pd.to_datetime(process["timestamp"])
    process["rss_kib"] = pd.to_numeric(process["rss_kib"], errors="coerce")
    is_env = process["args"].fillna("").str.contains("EnvWorker", regex=False)
    env = (
        process[is_env]
        .groupby("timestamp", as_index=False)["rss_kib"]
        .sum()
        .rename(columns={"rss_kib": "env_rss_kib"})
        .sort_values("timestamp")
    )
    env["elapsed_s"] = (env["timestamp"] - formal["timestamp"].min()).dt.total_seconds()
    env["env_rss_gib"] = env["env_rss_kib"] / (1024**2)
    formal = pd.merge_asof(
        formal.sort_values("elapsed_s"),
        env[["elapsed_s", "env_rss_gib"]].sort_values("elapsed_s"),
        on="elapsed_s",
        direction="nearest",
        tolerance=15,
    )
    formal["run"] = "DVAC formal (step22 snapshot)"
    return formal, raw


def load_baseline() -> tuple[pd.DataFrame, dict[str, object]]:
    baseline = pd.read_csv(BASELINE_RESOURCE_CSV)
    baseline["timestamp"] = pd.to_datetime(baseline["timestamp"])
    numeric = [column for column in baseline.columns if column not in {"timestamp", "top_command"}]
    baseline[numeric] = baseline[numeric].apply(pd.to_numeric, errors="coerce")
    baseline = baseline.sort_values("timestamp").reset_index(drop=True)
    baseline["elapsed_s"] = (
        baseline["timestamp"] - baseline.iloc[0]["timestamp"]
    ).dt.total_seconds()
    baseline["cgroup_current_gib"] = baseline["cgroup_ram_mb"] / MIB_PER_GIB
    baseline["gpu0_memory_gib"] = baseline["gpu0_memory_mb"] / MIB_PER_GIB
    baseline["gpu1_memory_gib"] = baseline["gpu1_memory_mb"] / MIB_PER_GIB
    baseline["env_rss_gib"] = baseline["env_rss_mb"] / MIB_PER_GIB
    baseline["run"] = "Successful GRPO baseline"
    analysis = json.loads(BASELINE_ANALYSIS_JSON.read_text(encoding="utf-8"))
    return baseline, analysis


def ten_minute_bins(frame: pd.DataFrame, run_label: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["bin_index"] = (frame["elapsed_s"] // BIN_SECONDS).astype(int)
    rows: list[dict[str, object]] = []
    for bin_index, group in frame.groupby("bin_index"):
        group = group.sort_values("elapsed_s")
        row: dict[str, object] = {
            "run": run_label,
            "bin_index": int(bin_index),
            "bin_start_h": fnum(bin_index * BIN_SECONDS / 3600.0),
            "bin_end_h": fnum((bin_index + 1) * BIN_SECONDS / 3600.0),
            "samples": int(len(group)),
        }
        for column in [
            "cgroup_current_gib",
            "gpu0_memory_gib",
            "gpu1_memory_gib",
            "env_rss_gib",
        ]:
            values = group[column].dropna()
            row[f"{column}_median"] = fnum(values.median()) if len(values) else None
            row[f"{column}_peak"] = fnum(values.max()) if len(values) else None
            row[f"{column}_last"] = fnum(values.iloc[-1]) if len(values) else None
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary(
    formal: pd.DataFrame,
    baseline: pd.DataFrame,
    baseline_analysis: dict[str, object],
) -> dict[str, object]:
    formal_duration_h = float(formal["elapsed_s"].max()) / 3600.0
    matched_seconds = float(formal["elapsed_s"].max())
    formal_window = formal[formal["elapsed_s"] <= FIRST_WINDOW_HOURS * 3600].copy()
    baseline_window = baseline[baseline["elapsed_s"] <= FIRST_WINDOW_HOURS * 3600].copy()
    baseline_matched = baseline[baseline["elapsed_s"] <= matched_seconds].copy()
    baseline_full = baseline.copy()

    metric_columns = [
        "cgroup_current_gib",
        "gpu0_memory_gib",
        "gpu1_memory_gib",
        "env_rss_gib",
    ]
    comparison = {
        "formal_observed_window": {
            column: stat_block(formal_window, column) for column in metric_columns
        },
        "baseline_first_10h": {
            column: stat_block(baseline_window, column) for column in metric_columns
        },
        "baseline_matched_elapsed": {
            "elapsed_h": fnum(matched_seconds / 3600.0),
            **{column: stat_block(baseline_matched, column) for column in metric_columns},
        },
        "baseline_full_run": {
            "duration_h": fnum(baseline_full["elapsed_s"].max() / 3600.0),
            **{column: stat_block(baseline_full, column) for column in metric_columns},
        },
    }

    formal_cg = comparison["formal_observed_window"]["cgroup_current_gib"]
    baseline_cg = comparison["baseline_matched_elapsed"]["cgroup_current_gib"]
    formal_env = comparison["formal_observed_window"]["env_rss_gib"]
    baseline_env = comparison["baseline_matched_elapsed"]["env_rss_gib"]
    deltas = {
        "formal_minus_baseline_matched": {
            "cgroup_peak_gib": fnum(formal_cg["peak"] - baseline_cg["peak"]),
            "cgroup_end_gib": fnum(formal_cg["end"] - baseline_cg["end"]),
            "cgroup_growth_from_own_start_gib": fnum(
                formal_cg["growth_end_minus_start"] - baseline_cg["growth_end_minus_start"]
            ),
            "cgroup_peak_growth_from_own_start_gib": fnum(
                formal_cg["peak_growth_from_start"] - baseline_cg["peak_growth_from_start"]
            ),
            "env_rss_peak_gib": fnum(formal_env["peak"] - baseline_env["peak"]),
            "env_rss_end_gib": fnum(formal_env["end"] - baseline_env["end"]),
            "gpu0_peak_gib": fnum(
                comparison["formal_observed_window"]["gpu0_memory_gib"]["peak"]
                - comparison["baseline_matched_elapsed"]["gpu0_memory_gib"]["peak"]
            ),
            "gpu1_peak_gib": fnum(
                comparison["formal_observed_window"]["gpu1_memory_gib"]["peak"]
                - comparison["baseline_matched_elapsed"]["gpu1_memory_gib"]["peak"]
            ),
        }
    }
    peak_authority = baseline_analysis["peak"]
    analysis_peak_check = {
        "analysis_json_target_completed": int(max(m["step"] for m in baseline_analysis["metrics"])),
        "analysis_json_target_step": int(max(m["target_step"] for m in baseline_analysis["metrics"])),
        "cgroup_peak_gib": fnum(peak_authority["peak_ram_mb"] / MIB_PER_GIB),
        "gpu0_peak_gib": fnum(peak_authority["peak_gpu0_mb"] / MIB_PER_GIB),
        "gpu1_peak_gib": fnum(peak_authority["peak_gpu1_mb"] / MIB_PER_GIB),
        "env_rss_peak_gib": fnum(peak_authority["peak_env_rss_mb"] / MIB_PER_GIB),
        "cgroup_oom": int(peak_authority["cgroup_oom"]),
        "cgroup_oom_kill": int(peak_authority["cgroup_oom_kill"]),
    }
    return {
        "sources": {
            "formal_resources": str(FORMAL_RESOURCE_CSV),
            "formal_process_rss": str(FORMAL_PROCESS_TSV),
            "baseline_resources": str(BASELINE_RESOURCE_CSV),
            "baseline_analysis": str(BASELINE_ANALYSIS_JSON),
        },
        "alignment": {
            "x_axis": "hours since each resource monitor's first sample",
            "formal_observed_hours": fnum(formal_duration_h),
            "comparison_window_hours": FIRST_WINDOW_HOURS,
            "cgroup_comparability": "Same memory.current-style cgroup total and 240 GiB limit; formal bytes were converted to GiB, baseline monitor MiB to GiB.",
            "gpu_comparability": "Both use nvidia-smi per-GPU used memory; formal samples are long-form per GPU, baseline samples are wide-form.",
            "env_rss_comparability": "Both are aggregate EnvWorker RSS. Formal is reconstructed from process_rss.tsv sampled about every 11 s; baseline env_rss_mb was emitted directly in its about-2 s resource monitor. RSS can double-count shared pages.",
            "time_alignment_limit": "Elapsed zero is monitor start, not optimizer-step alignment. Formal snapshot ends before 10 h; baseline first-10-h and matched-9.862-h views are both reported.",
            "baseline_start_offset": "Baseline cgroup starts materially higher because it includes pre-existing container memory; compare both raw totals and growth relative to each run's first sample.",
        },
        "comparison": comparison,
        "deltas": deltas,
        "baseline_analysis_json_peak_check": analysis_peak_check,
        "interpretation": {
            "gpu": "Formal per-GPU peaks are close to the successful baseline and remain far below 80 GiB capacity.",
            "cgroup": "At matched elapsed time, formal raw cgroup total is lower because it began with less pre-existing container memory; growth from each run's own start is nearly identical. Both runs show strong host-memory growth, and the successful baseline later operated near the 240 GiB limit.",
            "env_rss": "Formal aggregate EnvWorker RSS is already close to the baseline's matched first-10-h level and below the baseline full-run peak; this is the main growth source to keep watching.",
            "causal_boundary": "This observational comparison does not isolate DVAC overhead because monitor start state, exact phase timing, and instrumentation differ.",
        },
    }


def draw_dashed(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    fill: str,
    width: int = 3,
    dash: int = 12,
    gap: int = 8,
) -> None:
    if len(points) < 2:
        return
    for p0, p1 in zip(points[:-1], points[1:]):
        x0, y0 = p0
        x1, y1 = p1
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length <= 0:
            continue
        ux, uy = dx / length, dy / length
        position = 0.0
        while position < length:
            end = min(position + dash, length)
            draw.line(
                (
                    x0 + ux * position,
                    y0 + uy * position,
                    x0 + ux * end,
                    y0 + uy * end,
                ),
                fill=fill,
                width=width,
            )
            position += dash + gap


def draw_overlay_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    series: list[dict[str, object]],
    y_min: float,
    y_max: float,
    y_ticks: list[float],
    y_unit: str,
    reference_lines: list[tuple[float, str, str]] | None = None,
) -> None:
    x0, y0, x1, y1 = box
    left, right, top, bottom = x0 + 76, x1 - 24, y0 + 66, y1 - 44
    small = get_font(14)
    label = get_font(16)
    grid, axis = "#e7ecf2", "#617086"
    for tick in y_ticks:
        yy = bottom - (tick - y_min) / (y_max - y_min) * (bottom - top)
        draw.line((left, yy, right, yy), fill=grid, width=1)
        text = f"{tick:g}"
        tw = draw.textbbox((0, 0), text, font=small)[2]
        draw.text((left - tw - 10, yy - 8), text, font=small, fill=axis)
    for hour in [0, 2, 4, 6, 8, 10]:
        xx = left + hour / FIRST_WINDOW_HOURS * (right - left)
        draw.line((xx, top, xx, bottom), fill=grid, width=1)
        text = f"{hour}h"
        tw = draw.textbbox((0, 0), text, font=small)[2]
        draw.text((xx - tw / 2, bottom + 10), text, font=small, fill=axis)
    draw.line((left, top, left, bottom), fill=axis, width=2)
    draw.line((left, bottom, right, bottom), fill=axis, width=2)
    draw.text((x0 + 17, top - 2), y_unit, font=small, fill=axis)
    if reference_lines:
        for value, text, color in reference_lines:
            if y_min <= value <= y_max:
                yy = bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
                draw.line((left, yy, right, yy), fill=color, width=2)
                tw = draw.textbbox((0, 0), text, font=small)[2]
                draw.rectangle((right - tw - 9, yy - 18, right, yy + 1), fill="#ffffff")
                draw.text((right - tw - 4, yy - 17), text, font=small, fill=color)

    legend_x = left
    for item in series:
        color = str(item["color"])
        dashed = bool(item.get("dashed", False))
        if dashed:
            draw.line((legend_x, y0 + 47, legend_x + 12, y0 + 47), fill=color, width=3)
            draw.line((legend_x + 20, y0 + 47, legend_x + 32, y0 + 47), fill=color, width=3)
        else:
            draw.line((legend_x, y0 + 47, legend_x + 32, y0 + 47), fill=color, width=4)
        text = str(item["label"])
        draw.text((legend_x + 39, y0 + 36), text, font=label, fill="#26354b")
        legend_x += 39 + draw.textbbox((0, 0), text, font=label)[2] + 24

    for item in series:
        xs = np.asarray(item["x"], dtype=float)
        ys = np.asarray(item["y"], dtype=float)
        valid = np.isfinite(xs) & np.isfinite(ys) & (xs <= FIRST_WINDOW_HOURS)
        xs, ys = xs[valid], ys[valid]
        stride = max(1, math.ceil(len(xs) / 1800))
        xs, ys = xs[::stride], ys[::stride]
        points = []
        for xv, yv in zip(xs, ys):
            xx = left + xv / FIRST_WINDOW_HOURS * (right - left)
            yy = bottom - (yv - y_min) / (y_max - y_min) * (bottom - top)
            points.append((xx, max(top, min(bottom, yy))))
        if bool(item.get("dashed", False)):
            draw_dashed(draw, points, str(item["color"]), width=3)
        elif len(points) >= 2:
            draw.line(points, fill=str(item["color"]), width=3, joint="curve")


def render_plot(formal: pd.DataFrame, baseline: pd.DataFrame, summary: dict[str, object]) -> None:
    width, height = 1800, 1260
    image = Image.new("RGB", (width, height), "#f4f7fb")
    draw = ImageDraw.Draw(image)
    draw.text((48, 28), "DVAC formal vs successful GRPO: first 10 hours", font=get_font(34, True), fill="#111b2d")
    draw.text(
        (48, 77),
        "Elapsed time starts at each resource monitor's first sample; solid = DVAC formal, dashed = successful GRPO baseline.",
        font=get_font(20),
        fill="#526176",
    )

    panels = [
        (40, 122, 1760, 465),
        (40, 485, 1760, 828),
        (40, 848, 1760, 1191),
    ]
    draw_panel(draw, panels[0], "Cgroup current memory")
    draw_panel(draw, panels[1], "Per-GPU memory")
    draw_panel(draw, panels[2], "Aggregate EnvWorker RSS")

    formal_x = formal["elapsed_s"].to_numpy(dtype=float) / 3600.0
    baseline_x = baseline["elapsed_s"].to_numpy(dtype=float) / 3600.0
    draw_overlay_chart(
        draw,
        panels[0],
        [
            {"x": formal_x, "y": formal["cgroup_current_gib"], "label": "DVAC formal", "color": "#7c3aed"},
            {"x": baseline_x, "y": baseline["cgroup_current_gib"], "label": "GRPO baseline", "color": "#64748b", "dashed": True},
        ],
        0,
        240,
        [0, 60, 120, 180, 240],
        "GiB",
        [(240, "240 GiB limit", "#b91c1c")],
    )
    draw_overlay_chart(
        draw,
        panels[1],
        [
            {"x": formal_x, "y": formal["gpu0_memory_gib"], "label": "DVAC GPU0", "color": "#2563eb"},
            {"x": formal_x, "y": formal["gpu1_memory_gib"], "label": "DVAC GPU1", "color": "#ea580c"},
            {"x": baseline_x, "y": baseline["gpu0_memory_gib"], "label": "GRPO GPU0", "color": "#2563eb", "dashed": True},
            {"x": baseline_x, "y": baseline["gpu1_memory_gib"], "label": "GRPO GPU1", "color": "#ea580c", "dashed": True},
        ],
        0,
        80,
        [0, 20, 40, 60, 80],
        "GiB",
        [(80, "80 GiB capacity", "#9aa6b5")],
    )
    baseline_full_peak = summary["comparison"]["baseline_full_run"]["env_rss_gib"]["peak"]
    draw_overlay_chart(
        draw,
        panels[2],
        [
            {"x": formal_x, "y": formal["env_rss_gib"], "label": "DVAC formal", "color": "#7c3aed"},
            {"x": baseline_x, "y": baseline["env_rss_gib"], "label": "GRPO baseline", "color": "#64748b", "dashed": True},
        ],
        0,
        160,
        [0, 40, 80, 120, 160],
        "GiB",
        [(float(baseline_full_peak), f"baseline full peak {baseline_full_peak:.1f}", "#9a3412")],
    )

    delta = summary["deltas"]["formal_minus_baseline_matched"]
    formal_cg = summary["comparison"]["formal_observed_window"]["cgroup_current_gib"]
    baseline_cg = summary["comparison"]["baseline_matched_elapsed"]["cgroup_current_gib"]
    baseline_final = summary["comparison"]["baseline_full_run"]["cgroup_current_gib"]
    footer = (
        f"At matched 9.86 h: cgroup end DVAC {formal_cg['end']:.1f} vs GRPO {baseline_cg['end']:.1f} GiB; "
        f"growth from each start differs by {delta['cgroup_growth_from_own_start_gib']:+.1f} GiB.  "
        f"Successful GRPO full-run cgroup peak {baseline_final['peak']:.1f} GiB.  "
        "RSS sampling differs; cgroup is the authoritative total."
    )
    draw.text((48, 1217), footer, font=get_font(17), fill="#526176")
    image.save(PNG_OUT, optimize=True)


def main() -> None:
    formal, _ = load_formal()
    baseline, baseline_analysis = load_baseline()
    binned = pd.concat(
        [
            ten_minute_bins(formal, "dvac_formal"),
            ten_minute_bins(baseline, "successful_grpo_baseline"),
        ],
        ignore_index=True,
    )
    binned.to_csv(CSV_OUT, index=False)
    summary = build_summary(formal, baseline, baseline_analysis)
    JSON_OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    render_plot(formal, baseline, summary)
    print(CSV_OUT)
    print(JSON_OUT)
    print(PNG_OUT)


if __name__ == "__main__":
    main()
