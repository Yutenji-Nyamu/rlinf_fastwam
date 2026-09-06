#!/usr/bin/env python3
"""Build a compact, checkpoint-free evidence package for the OGPO 35k run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import shutil
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
STEP_RE = re.compile(r"Global Step:\s*([0-9]+)/([0-9]+)")
PAIR_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_./]*)=([-+0-9.eE]+)")
ELAPSED_RE = re.compile(r"Elapsed:\s*([0-9:]+)")


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def describe(values: list[float]) -> dict[str, float | None]:
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
    for wave, match in enumerate(matches, start=1):
        end = matches[wave].start() if wave < len(matches) else len(text)
        block = text[match.start() : end]
        values: dict[str, list[float]] = defaultdict(list)
        for key, raw in PAIR_RE.findall(block):
            try:
                values[key].append(float(raw))
            except ValueError:
                continue
        record: dict[str, float] = {
            "wave": float(wave),
            "rows": float(match.group(1)),
            "row_budget": float(match.group(2)),
        }
        elapsed = ELAPSED_RE.search(block)
        if elapsed:
            parts = [int(item) for item in elapsed.group(1).split(":")]
            seconds = 0
            for part in parts:
                seconds = seconds * 60 + part
            record["elapsed_seconds"] = float(seconds)
        for key, items in values.items():
            record["step_seconds" if key == "step" else key] = items[-1]
        successes = values.get("success_once", [])
        if successes:
            record["train_success_once"] = successes[0]
        if len(successes) > 1:
            record["eval_success_once"] = successes[-1]
        records.append(record)
    return records


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
        "disk_available_bytes",
        "gpu0_used_mib",
        "gpu1_used_mib",
        "gpu0_util_pct",
        "gpu1_util_pct",
        "gpu0_power_w",
        "gpu1_power_w",
        "matched_total_rss_kib",
        "compute_process_count",
    ]
    summary = {
        key: describe([row[key] for row in rows if key in row and row[key] >= 0])
        for key in keys
    }
    times = [row["unix_time"] for row in rows if "unix_time" in row]
    gaps = [b - a for a, b in zip(times, times[1:]) if b >= a]
    summary["meta"] = {
        "rows": len(rows),
        "first_unix": times[0] if times else None,
        "last_unix": times[-1] if times else None,
        "duration_seconds": times[-1] - times[0] if len(times) > 1 else 0,
        "cadence_p50_seconds": percentile(gaps, 0.50),
        "cadence_p95_seconds": percentile(gaps, 0.95),
        "max_gap_seconds": max(gaps, default=None),
        "oom_max": max((row.get("cgroup_oom_events", 0) for row in rows), default=None),
        "oom_kill_max": max(
            (row.get("cgroup_oom_kill_events", 0) for row in rows), default=None
        ),
    }
    return rows, summary


def downsample(rows: list[dict[str, float]], maximum: int = 700) -> list[dict[str, float]]:
    if len(rows) <= maximum:
        return rows
    indices = {round(i * (len(rows) - 1) / (maximum - 1)) for i in range(maximum)}
    return [rows[index] for index in sorted(indices)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def series_path(
    points: list[tuple[float, float]],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    width: float = 920,
    height: float = 220,
    left: float = 62,
    top: float = 18,
) -> str:
    if not points:
        return ""
    inner_w = width - left - 22
    inner_h = height - top - 42

    def xy(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        px = left + (x - x_min) / max(x_max - x_min, 1e-12) * inner_w
        py = top + (y_max - y) / max(y_max - y_min, 1e-12) * inner_h
        return px, py

    return " ".join(
        ("M" if index == 0 else "L") + f"{px:.2f},{py:.2f}"
        for index, (px, py) in enumerate(xy(point) for point in points)
    )


def moving_average(points: list[tuple[float, float]], window: int = 5) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for index, (x, _) in enumerate(points):
        start = max(0, index - window + 1)
        result.append((x, statistics.fmean(value for _, value in points[start : index + 1])))
    return result


def chart_svg(
    *,
    label: str,
    description: str,
    x_max: float,
    y_min: float,
    y_max: float,
    y_ticks: list[float],
    series: list[tuple[str, str, list[tuple[float, float]], str]],
    suffix: str = "",
    reference: float | None = None,
    points: list[tuple[float, float, str]] | None = None,
) -> str:
    width, height, left, top = 920, 220, 62, 18
    inner_w = width - left - 22
    inner_h = height - top - 42

    def x_pos(value: float) -> float:
        return left + value / max(x_max, 1e-12) * inner_w

    def y_pos(value: float) -> float:
        return top + (y_max - value) / max(y_max - y_min, 1e-12) * inner_h

    grid = []
    for tick in y_ticks:
        y = y_pos(tick)
        grid.append(
            f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + inner_w}" y2="{y:.2f}" />'
        )
        grid.append(
            f'<text class="axis" x="{left - 9}" y="{y + 4:.2f}" text-anchor="end">{tick:g}{html.escape(suffix)}</text>'
        )
    for tick in [0, x_max / 4, x_max / 2, x_max * 3 / 4, x_max]:
        x = x_pos(tick)
        grid.append(
            f'<text class="axis" x="{x:.2f}" y="{height - 8}" text-anchor="middle">{tick / 1000:g}k</text>'
        )
    paths = []
    legend = []
    for name, css_class, values, dash in series:
        path = series_path(values, 0, x_max, y_min, y_max)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        paths.append(f'<path class="line {css_class}" d="{path}"{dash_attr} />')
        legend.append(
            f'<span><i class="swatch {css_class}"></i>{html.escape(name)}</span>'
        )
    if reference is not None:
        y = y_pos(reference)
        paths.append(
            f'<line class="reference" x1="{left}" y1="{y:.2f}" x2="{left + inner_w}" y2="{y:.2f}" />'
        )
    marks = []
    for x_value, y_value, text_value in points or []:
        x, y = x_pos(x_value), y_pos(y_value)
        marks.append(f'<circle class="point series-2" cx="{x:.2f}" cy="{y:.2f}" r="5" />')
        marks.append(
            f'<text class="mark-label" x="{x:.2f}" y="{max(12, y - 9):.2f}" text-anchor="middle">{html.escape(text_value)}</text>'
        )
    return f"""
<section class="plot-section">
  <div class="plot-label"><span>{html.escape(label)}</span><span class="legend">{''.join(legend)}</span></div>
  <svg class="plot" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(description)}">
    <title>{html.escape(label)}</title><desc>{html.escape(description)}</desc>
    {''.join(grid)}
    <line class="axis-line" x1="{left}" y1="{top + inner_h}" x2="{left + inner_w}" y2="{top + inner_h}" />
    {''.join(paths)}
    {''.join(marks)}
  </svg>
</section>"""


def base_fragment(root_id: str, body: str) -> str:
    return f"""<div id="{root_id}" class="ogpo-viz">
<style>
#{root_id} {{ color: var(--foreground); width: 100%; }}
#{root_id} .plot-section {{ margin: 0 0 18px; }}
#{root_id} .plot-label {{ display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 8px; font-weight: 500; margin-bottom: 4px; }}
#{root_id} .legend {{ display: flex; flex-wrap: wrap; gap: 12px; color: var(--muted-foreground); font-weight: 400; }}
#{root_id} .legend span {{ display: inline-flex; align-items: center; gap: 5px; }}
#{root_id} .swatch {{ display: inline-block; width: 18px; height: 3px; background: var(--viz-series-1); }}
#{root_id} .swatch.series-2 {{ background: var(--viz-series-2); }}
#{root_id} .swatch.series-3 {{ background: var(--viz-series-3); }}
#{root_id} .plot {{ width: 100%; height: auto; display: block; overflow: visible; }}
#{root_id} .grid {{ stroke: var(--border); stroke-width: 1; }}
#{root_id} .axis-line {{ stroke: var(--muted-foreground); stroke-width: 1; }}
#{root_id} .axis, #{root_id} .mark-label {{ fill: var(--muted-foreground); font-size: 12px; }}
#{root_id} .mark-label {{ fill: var(--foreground); font-weight: 500; }}
#{root_id} .line {{ fill: none; stroke: var(--viz-series-1); stroke-width: 2.5; vector-effect: non-scaling-stroke; }}
#{root_id} .line.series-2 {{ stroke: var(--viz-series-2); }}
#{root_id} .line.series-3 {{ stroke: var(--viz-series-3); }}
#{root_id} .point {{ fill: var(--background); stroke: var(--viz-series-1); stroke-width: 3; vector-effect: non-scaling-stroke; }}
#{root_id} .point.series-2 {{ stroke: var(--viz-series-2); }}
#{root_id} .reference {{ stroke: var(--muted-foreground); stroke-width: 1.5; stroke-dasharray: 5 5; vector-effect: non-scaling-stroke; }}
#{root_id} .allocation {{ display: grid; grid-template-columns: 68.9fr 28.5fr 1.9fr .7fr; height: 28px; margin: 6px 0; overflow: hidden; }}
#{root_id} .allocation span {{ min-width: 2px; background: var(--viz-series-1); }}
#{root_id} .allocation span:nth-child(2) {{ background: var(--viz-series-2); }}
#{root_id} .allocation span:nth-child(3) {{ background: var(--viz-series-3); }}
#{root_id} .allocation span:nth-child(4) {{ background: var(--muted); }}
#{root_id} .allocation-labels {{ display: flex; flex-wrap: wrap; gap: 12px; color: var(--muted-foreground); }}
#{root_id} .allocation-labels b {{ color: var(--foreground); font-weight: 500; }}
@media (max-width: 520px) {{ #{root_id} .plot-label {{ align-items: flex-start; }} #{root_id} .legend {{ width: 100%; }} }}
</style>
{body}
</div>
"""


def build_fragments(
    records: list[dict[str, float]], resources: list[dict[str, float]], visual_dir: Path, package_visuals: Path
) -> list[Path]:
    x_max = 35000.0
    train_points = [
        (record["rows"], record.get("train_success_once", 0.0) * 100)
        for record in records
    ]
    train_avg = moving_average(train_points, 5)
    eval_points = [(0.0, 5.0, "5%"), (20081.0, 35.0, "35%"), (35000.0, 5.0, "5%")]
    ratio = [(r["rows"], r["ogpo/ratio"]) for r in records if "ogpo/ratio" in r]
    bc_loss = [(r["rows"], r["ogpo/bc_loss"]) for r in records if "ogpo/bc_loss" in r]
    critic_loss = [
        (r["rows"], min(r["ogpo/critic_loss"], 0.12))
        for r in records
        if "ogpo/critic_loss" in r
    ]

    success_chart = chart_svg(
        label="Policy success vs replay rows",
        description="Noisy per-wave train success, five-wave moving average, and three fixed-policy evaluation points.",
        x_max=x_max,
        y_min=0,
        y_max=70,
        y_ticks=[0, 20, 40, 60],
        suffix="%",
        series=[
            ("train wave", "series-1", train_points, "2 5"),
            ("5-wave mean", "series-3", train_avg, ""),
        ],
        points=eval_points,
    )
    ratio_chart = chart_svg(
        label="Online / EMA same-chain ratio",
        description="Mean probability ratio per update burst. It fell early and recovered to 0.942, still below the 1.0 identity line.",
        x_max=x_max,
        y_min=0,
        y_max=1.05,
        y_ticks=[0, 0.25, 0.5, 0.75, 1.0],
        series=[("ratio", "series-1", ratio, "")],
        reference=1.0,
    )
    loss_chart = chart_svg(
        label="BC and critic losses after warmup",
        description="Success behavior-cloning loss and critic loss. The first critic spike is capped at 0.12 so the stable region remains visible.",
        x_max=x_max,
        y_min=0,
        y_max=0.12,
        y_ticks=[0, 0.03, 0.06, 0.09, 0.12],
        series=[
            ("success BC", "series-1", bc_loss, ""),
            ("critic (first spike capped)", "series-2", critic_loss, ""),
        ],
    )
    training_body = success_chart + ratio_chart + loss_chart
    training_fragment = base_fragment("ogpo-training-35k", training_body)

    sampled = downsample(resources, 700)
    # Preserve narrow peaks that uniform downsampling can otherwise miss, especially
    # the short final-checkpoint serialization peak.
    extrema = [
        max(resources, key=lambda row: row.get("cgroup_current_bytes", -1)),
        max(resources, key=lambda row: row.get("gpu0_used_mib", -1)),
        max(resources, key=lambda row: row.get("gpu1_used_mib", -1)),
    ]
    sampled = sorted(
        {row["unix_time"]: row for row in [*sampled, *extrema]}.values(),
        key=lambda row: row["unix_time"],
    )
    start = sampled[0]["unix_time"]
    duration_hours = (sampled[-1]["unix_time"] - start) / 3600
    gpu0 = [((r["unix_time"] - start) / 3600, r["gpu0_used_mib"] / 1024) for r in sampled]
    gpu1 = [((r["unix_time"] - start) / 3600, r["gpu1_used_mib"] / 1024) for r in sampled]
    gpu0_util = [((r["unix_time"] - start) / 3600, r["gpu0_util_pct"]) for r in sampled]
    gpu1_util = [((r["unix_time"] - start) / 3600, r["gpu1_util_pct"]) for r in sampled]
    cgroup = [((r["unix_time"] - start) / 3600, r["cgroup_current_bytes"] / 2**30) for r in sampled]

    def time_chart(
        label: str,
        description: str,
        y_min: float,
        y_max: float,
        y_ticks: list[float],
        series: list[tuple[str, str, list[tuple[float, float]], str]],
        suffix: str,
    ) -> str:
        scaled = [
            (name, css, [(x / duration_hours * 35000, y) for x, y in values], dash)
            for name, css, values, dash in series
        ]
        rendered = chart_svg(
            label=label,
            description=description,
            x_max=35000,
            y_min=y_min,
            y_max=y_max,
            y_ticks=y_ticks,
            series=scaled,
            suffix=suffix,
        )
        rendered = rendered.replace(">0k<", ">0h<")
        rendered = rendered.replace(">8.75k<", f">{duration_hours/4:.1f}h<")
        rendered = rendered.replace(">17.5k<", f">{duration_hours/2:.1f}h<")
        rendered = rendered.replace(">26.25k<", f">{duration_hours*3/4:.1f}h<")
        rendered = rendered.replace(">35k<", f">{duration_hours:.1f}h<")
        return rendered

    allocation = """
<section class="plot-section">
  <div class="plot-label"><span>Measured wall-time allocation</span></div>
  <div class="allocation" role="img" aria-label="Training 68.9 percent, rollout 28.5 percent, evaluation 1.9 percent, other 0.7 percent"><span></span><span></span><span></span><span></span></div>
  <div class="allocation-labels"><span><b>68.9%</b> paired updates</span><span><b>28.5%</b> rollout</span><span><b>1.9%</b> eval</span><span><b>0.7%</b> other</span></div>
</section>
"""
    gpu_memory = time_chart(
        "Physical GPU memory",
        "Physical memory usage on both A800 GPUs over the complete run.",
        0,
        64,
        [0, 16, 32, 48, 64],
        [("GPU 0", "series-1", gpu0, ""), ("GPU 1", "series-2", gpu1, "")],
        " GiB",
    )
    gpu_util = time_chart(
        "GPU utilization",
        "One-second GPU utilization on both cards.",
        0,
        100,
        [0, 25, 50, 75, 100],
        [("GPU 0", "series-1", gpu0_util, ""), ("GPU 1", "series-2", gpu1_util, "")],
        "%",
    )
    memory = time_chart(
        "Container memory (checkpoint peak included)",
        "Cgroup memory rose with replay and peaked during final checkpoint serialization.",
        0,
        240,
        [0, 60, 120, 180, 240],
        [("cgroup", "series-3", cgroup, "")],
        " GiB",
    )
    resource_fragment = base_fragment("ogpo-resources-35k", allocation + gpu_memory + gpu_util + memory)
    combined_fragment = base_fragment("ogpo-overview-35k", training_body + allocation + gpu_memory + memory)

    package_visuals.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        package_visuals / "training-fragment.html",
        package_visuals / "resources-fragment.html",
        visual_dir / "ogpo-formal-35k-overview.html",
    ]
    for path, text in zip(outputs, [training_fragment, resource_fragment, combined_fragment]):
        if any(token in text.lower() for token in ["<!doctype", "<html", "<head", "<body"]):
            raise ValueError(f"fragment contract violation: {path}")
        path.write_text(text, encoding="utf-8", newline="\n")
    return outputs


def copy_inputs(package: Path, immutable: Path, final: Path) -> None:
    files = {
        immutable / "resolved.yaml": package / "resolved.yaml",
        immutable / "source_config.yaml": package / "source_config.yaml",
        immutable / "exact_command.txt": package / "exact_command.txt",
        immutable / "run_provenance.tsv": package / "run_provenance.tsv",
        immutable / "stop_conditions.txt": package / "stop_conditions.txt",
        immutable / "started_at.txt": package / "started_at.txt",
        immutable / "resources_before.txt": package / "resources_before.txt",
        final / "finished_at.txt": package / "finished_at.txt",
        final / "exit_code.txt": package / "exit_code.txt",
        final / "metrics.log": package / "metrics.log",
        final / "driver.log": package / "driver.log",
        final / "resources_1s.csv": package / "resources_1s.csv",
        final / "resources_after.txt": package / "resources_after.txt",
        final / "tensorboard_config.yaml": package / "tensorboard_config.yaml",
        final / "events.out.tfevents.103684.0": package / "events.out.tfevents.103684.0",
        final / "checkpoint_complete.json": package / "checkpoint_complete.json",
    }
    package.mkdir(parents=True, exist_ok=True)
    for source, destination in files.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination)


def write_tables(package: Path, records: list[dict[str, float]], resource_summary: dict) -> None:
    fields = [
        "wave",
        "rows",
        "elapsed_seconds",
        "train_success_once",
        "eval_success_once",
        "ogpo/actor_updates",
        "ogpo/critic_updates",
        "ogpo/updates_run",
        "ogpo/ratio",
        "ogpo/actor_loss",
        "ogpo/actor_grad_norm",
        "ogpo/bc_loss",
        "ogpo/critic_loss",
        "ogpo/critic_grad_norm",
        "ogpo/q_mean",
        "ogpo/td_target_mean",
        "generate_rollouts",
        "actor/run_training",
        "eval",
        "step_seconds",
    ]
    with (package / "metrics_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    with (package / "evaluation_points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rows", "successes", "episodes", "success_rate"])
        writer.writerows([[0, 1, 20, 0.05], [20081, 7, 20, 0.35], [35000, 1, 20, 0.05]])
    (package / "resource_summary.json").write_text(
        json.dumps(resource_summary, indent=2, sort_keys=True), encoding="utf-8", newline="\n"
    )


def build_summary(records: list[dict[str, float]], resource_summary: dict) -> dict:
    return {
        "run": {
            "status": "complete",
            "exit_code": 0,
            "started_at": "2026-08-07T23:13:50+08:00",
            "finished_at": "2026-08-08T11:37:32+08:00",
            "wall_seconds": 44622,
            "rows": 35000,
            "warmup_rows": 10000,
            "paired_updates": 2500,
            "waves": 26,
            "train_episodes": 208,
            "train_successes": 57,
            "train_success_rate": 57 / 208,
            "full_200_step_episode_equivalents": 175,
        },
        "fixed_policy_eval": [
            {"rows": 0, "successes": 1, "episodes": 20, "success_rate": 0.05},
            {"rows": 20081, "successes": 7, "episodes": 20, "success_rate": 0.35},
            {"rows": 35000, "successes": 1, "episodes": 20, "success_rate": 0.05},
        ],
        "final_optimization": {
            "actor_loss": 4.892e-5,
            "actor_grad_norm": 0.1829,
            "success_bc_loss": 0.02019,
            "online_ema_ratio": 0.94224,
            "critic_loss": 0.009527,
            "critic_grad_norm": 0.3047,
            "q_mean": 0.04582,
            "td_target_mean": 0.03996,
        },
        "wall_allocation": {
            "rollout_seconds": 12714.3,
            "paired_training_seconds": 30732.518,
            "evaluation_seconds_approx": 833.0,
            "other_seconds_approx": 342.2,
        },
        "resources": {
            "gpu0_peak_mib": resource_summary["gpu0_used_mib"]["max"],
            "gpu1_peak_mib": resource_summary["gpu1_used_mib"]["max"],
            "gpu0_mean_util_pct": resource_summary["gpu0_util_pct"]["mean"],
            "gpu1_mean_util_pct": resource_summary["gpu1_util_pct"]["mean"],
            "cgroup_peak_bytes": resource_summary["cgroup_current_bytes"]["max"],
            "oom_events": resource_summary["meta"]["oom_max"],
            "oom_kill_events": resource_summary["meta"]["oom_kill_max"],
        },
        "checkpoint": {
            "included_in_package": False,
            "complete_manifest_included": True,
            "checkpoint_total_bytes": 62526300538,
            "run_root_total_bytes": 62526543133,
            "dcp_shard_bytes": 11029316584,
            "dcp_metadata_bytes": 1149679,
            "rank_sidecar_bytes": 51495832880,
            "estimated_replay_payload_gib": 45.19,
            "note": "The large tensors stay on the server; this package includes only the completion manifest and inventory summary.",
        },
        "records": len(records),
    }


def write_docs(package: Path, summary: dict) -> None:
    (package / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8", newline="\n"
    )
    readme = """# OGPO pi0 RoboTwin formal 35k: compact evidence package

This package contains the high-information artifacts from the completed 35k run. It deliberately excludes the 58.23 GiB full checkpoint and all replay tensors.

## Result at a glance

- Exit 0 after 35,000 replay-valid primitive transitions, 10,000 warmup rows, and 2,500 paired actor+critic updates.
- 26 eight-environment waves = 208 simulated train episodes; 57 succeeded (27.4%).
- Fixed-policy evaluation: 1/20 at row 0, 7/20 at row 20,081, and 1/20 at row 35,000.
- Numerical training stayed finite, but the final policy lost the mid-run evaluation gain.
- Wall time was 12:23:42: paired updates 68.9%, rollout 28.5%, evaluation about 1.9%.
- GPU peaks were 57,231 and 57,676 MiB; no cgroup OOM or OOM-kill event occurred.

## Main files

- `SUMMARY.json`: machine-readable final outcome and checkpoint composition.
- `metrics_table.csv`: one row per 8-env train wave.
- `evaluation_points.csv`: the three fixed-policy evaluations.
- `metrics.log`, `driver.log`, TensorBoard event/config: original training evidence.
- `resources_1s.csv`, `resource_summary.json`: complete one-second resource trace and summary.
- `resolved.yaml`, `source_config.yaml`, `exact_command.txt`, `run_provenance.tsv`: exact launch provenance.
- `checkpoint_complete.json`, `checkpoint_inventory.tsv`: completion evidence without model/replay tensors.
- `visuals/*.png`: phone-readable training and resource plots.

## Interpretation boundary

The run establishes that the end-to-end OGPO+CA path executes and learns transiently. Three evaluation points are too sparse to locate the post-20k regression, and low critic TD loss alone does not establish reliable candidate ranking.
"""
    (package / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    glossary = """# Metric glossary

- `rows`: replay-valid primitive RoboTwin transitions, not action chunks or optimizer steps.
- `wave`: one outer runner cycle with eight parallel train episodes followed by the earned updates.
- `paired update`: one actor update plus one critic update.
- `train_success_once`: success fraction among the eight train episodes in one wave.
- `fixed-policy eval`: 20 evaluation episodes using the online actor at a frozen checkpoint in training time.
- `ratio`: mean same-chain probability ratio between online and EMA actors for a whole update burst.
- `success BC`: flow-matching behavior cloning on successful replay trajectories.
- `cgroup_current_bytes`: physical/accounted container memory, including page cache and checkpoint serialization transients.
"""
    (package / "METRIC_GLOSSARY.md").write_text(glossary, encoding="utf-8", newline="\n")
    inventory = """component\tbytes\tGiB\nDCP rank 0\t5514531924\t5.136\nDCP rank 1\t5514784660\t5.136\nDCP metadata\t1149679\t0.001071\nOGPO sidecar rank 0\t25273495010\t23.536\nOGPO sidecar rank 1\t26222337870\t24.421\ncomplete manifest\t1395\t0.0000013\ntotal checkpoint\t62526300538\t58.232\nrun-root non-checkpoint remainder\t242595\t0.000226\nrun root total\t62526543133\t58.232\n"""
    (package / "checkpoint_inventory.tsv").write_text(inventory, encoding="utf-8", newline="\n")


def finalize(package: Path, zip_path: Path) -> None:
    manifest = []
    for path in sorted(item for item in package.rglob("*") if item.is_file()):
        if path.name == "PACKAGE_MANIFEST.json":
            continue
        manifest.append(
            {
                "path": path.relative_to(package).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    (package / "PACKAGE_MANIFEST.json").write_text(
        json.dumps({"files": manifest}, indent=2, sort_keys=True), encoding="utf-8", newline="\n"
    )
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(item for item in package.rglob("*") if item.is_file()):
            archive.write(path, path.relative_to(package.parent))
    zip_path.with_suffix(zip_path.suffix + ".sha256").write_text(
        f"{sha256(zip_path)}  {zip_path.name}\n", encoding="utf-8", newline="\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--immutable", type=Path)
    parser.add_argument("--final", type=Path)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--visual-dir", type=Path)
    parser.add_argument("--zip", type=Path)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()

    if args.finalize:
        if args.zip is None:
            parser.error("--zip is required with --finalize")
        finalize(args.package, args.zip)
        return
    if args.immutable is None or args.final is None or args.visual_dir is None:
        parser.error("--immutable, --final, and --visual-dir are required")

    copy_inputs(args.package, args.immutable, args.final)
    records = parse_metric_tables(args.final / "metrics.log")
    resources, resource_summary = parse_resources(args.final / "resources_1s.csv")
    write_tables(args.package, records, resource_summary)
    summary = build_summary(records, resource_summary)
    write_docs(args.package, summary)
    build_fragments(records, resources, args.visual_dir, args.package / "visuals")


if __name__ == "__main__":
    main()
