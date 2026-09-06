from __future__ import annotations

import csv
import importlib.util
import json
import math
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(r"C:\Users\86136\Documents\rl")
EVIDENCE = ROOT / "docs" / "rlinf-shenzhen-pi0-ppo-rlt" / "evidence"
SNAPSHOT = EVIDENCE / "grpo_v2_live_step36_20260823"
BASE_SCRIPT = ROOT / "local_scripts" / "render_shenzhen_grpo_vs_ppo_20260823.py"
PPO_CSV = EVIDENCE / "ppo_formal100_console_scalars_step1_46_20260822.csv"
PPO_RESOURCE_CSV = EVIDENCE / "ppo_formal100_resource_discrete_samples_through_step46_20260822.csv"

SCALARS_OUT = SNAPSHOT / "grpo_console_scalars_step1_36.csv"
RESOURCE_STEP_OUT = SNAPSHOT / "grpo_resource_nearest_minute_by_completed_step.csv"
SUCCESS_OUT = SNAPSHOT / "grpo_vs_ppo_success_step1_36.png"
OPT_OUT = SNAPSHOT / "grpo_vs_ppo_optimization_timing_step1_36.png"
RESOURCE_OUT = SNAPSHOT / "grpo_resource_timeline_through_step36.png"
SUMMARY_OUT = SNAPSHOT / "summary_step36.json"


def load_base():
    spec = importlib.util.spec_from_file_location("grpo_plot_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B = load_base()


def trailing(values: list[float], window: int = 5) -> list[float | None]:
    return [
        statistics.mean(values[idx - window + 1 : idx + 1]) if idx >= window - 1 else None
        for idx in range(len(values))
    ]


def eval_steps(rows: list[dict]) -> set[int]:
    return {int(row["step"]) for row in rows if row["eval_success"] is not None}


def render_success(grpo: list[dict], ppo: list[dict]) -> None:
    steps = [float(row["step"]) for row in grpo]
    grpo_success = [float(row["train_success"]) for row in grpo]
    ppo_success = [float(row["train_success"]) for row in ppo]
    shaded = eval_steps(grpo) | eval_steps(ppo)
    img = Image.new("RGB", (1440, 1360), B.LIGHT)
    draw = B.add_header(
        img,
        "GRPO vs PPO: training success through Step 36",
        "Matched SZ-H100 budget: 4 GPUs and 512 trajectories per completed outer step",
    )
    B.line_panel(
        draw,
        (45, 145, 1395, 705),
        steps,
        [("GRPO raw", grpo_success, B.GRPO_LIGHT, 3), ("PPO raw", ppo_success, B.PPO_LIGHT, 3)],
        0.65,
        1.0,
        "Per-step training-rollout success",
        eval_steps=shaded,
    )
    plot = B.line_panel(
        draw,
        (45, 735, 1395, 1295),
        steps,
        [("GRPO trailing-5", trailing(grpo_success), B.GRPO, 7), ("PPO trailing-5", trailing(ppo_success), B.PPO, 7)],
        0.65,
        1.0,
        "Five-step mean and fixed-64 evaluation",
        eval_steps=shaded,
    )
    B.draw_eval_markers(
        draw,
        plot,
        steps,
        0.65,
        1.0,
        [(int(row["step"]), row["eval_success"]) for row in grpo],
        B.GRPO,
        "square",
    )
    B.draw_eval_markers(
        draw,
        plot,
        steps,
        0.65,
        1.0,
        [(int(row["step"]), row["eval_success"]) for row in ppo],
        B.PPO,
        "triangle",
    )
    draw.text((102, 1258), "Squares: GRPO fixed-64   Triangles: PPO fixed-64", fill=B.GRAY, font=B.F_SMALL)
    img.save(SUCCESS_OUT, optimize=True)


def render_optimization(grpo: list[dict], ppo: list[dict]) -> None:
    steps = [float(row["step"]) for row in grpo]
    shaded = eval_steps(grpo) | eval_steps(ppo)
    panels = [
        ("Approximate KL", "approx_kl", 0.0, 0.08),
        ("Clip fraction", "clip_fraction", 0.0, 0.14),
        ("Pre-clip gradient norm", "grad_norm", 0.0, 45.0),
        ("Whole-step wall time", "step_time_s", 0.0, 2100.0),
    ]
    img = Image.new("RGB", (1440, 1950), B.LIGHT)
    draw = B.add_header(
        img,
        "GRPO vs PPO: optimization and time through Step 36",
        "Complete-step console scalars; actor-only GRPO has no critic explained-variance series",
    )
    top, height, gap = 145, 410, 28
    for idx, (title, key, low, high) in enumerate(panels):
        B.line_panel(
            draw,
            (45, top + idx * (height + gap), 1395, top + idx * (height + gap) + height),
            steps,
            [("GRPO", [float(row[key]) for row in grpo], B.GRPO, 5), ("PPO", [float(row[key]) for row in ppo], B.PPO, 5)],
            low,
            high,
            title,
            eval_steps=shaded,
        )
    img.save(OPT_OUT, optimize=True)


def read_resources(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            gpu_mem = [float(raw[f"gpu{gpu}_used_mib"]) / 1024 for gpu in range(4, 8)]
            gpu_util = [float(raw[f"gpu{gpu}_util_pct"]) for gpu in range(4, 8)]
            rows.append(
                {
                    "timestamp": datetime.fromisoformat(raw["timestamp"]),
                    "cgroup_gib": float(raw["cgroup_memory_current_bytes"]) / 1024**3,
                    "host_available_gib": float(raw["host_mem_available_kib"]) / 1024**2,
                    "gpu_mean_gib": statistics.mean(gpu_mem),
                    "gpu_max_gib": max(gpu_mem),
                    "gpu_mean_util": statistics.mean(gpu_util),
                    "gpu_max_util": max(gpu_util),
                }
            )
    if len(rows) < 2:
        raise ValueError("resource.csv has fewer than two samples")
    return rows


def align_resources(grpo: list[dict], resources: list[dict]) -> list[dict]:
    start = resources[0]["timestamp"]
    output: list[dict] = []
    for metric in grpo:
        target = start + timedelta(seconds=float(metric["elapsed_s"]))
        nearest = min(resources, key=lambda row: abs((row["timestamp"] - target).total_seconds()))
        output.append(
            {
                "step": int(metric["step"]),
                "estimated_completion_utc": target.isoformat(),
                "nearest_resource_utc": nearest["timestamp"].isoformat(),
                "offset_seconds": int((nearest["timestamp"] - target).total_seconds()),
                "cgroup_gib": round(float(nearest["cgroup_gib"]), 3),
                "host_available_gib": round(float(nearest["host_available_gib"]), 3),
                "gpu_mean_gib": round(float(nearest["gpu_mean_gib"]), 3),
                "gpu_max_gib": round(float(nearest["gpu_max_gib"]), 3),
            }
        )
    if max(abs(int(row["offset_seconds"])) for row in output) > 31:
        raise ValueError("step/resource alignment exceeds one half-minute interval")
    return output


def render_resources(resources: list[dict], aligned: list[dict], ppo_resource: list[dict]) -> None:
    start = resources[0]["timestamp"]
    elapsed_h = [(row["timestamp"] - start).total_seconds() / 3600 for row in resources]
    last_h = elapsed_h[-1]
    hour_ticks = list(range(0, math.ceil(last_h) + 1, 2))
    if last_h not in hour_ticks:
        hour_ticks.append(last_h)
    img = Image.new("RGB", (1440, 2480), B.LIGHT)
    capture_cst = resources[-1]["timestamp"].astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M CST")
    draw = B.add_header(
        img,
        "GRPO resource timeline through Step 36",
        f"One-minute observer samples through {capture_cst}; instantaneous utilization is bursty during simulation",
    )
    panels = [
        (
            "Main-memory pressure",
            [("GRPO cgroup used", [float(row["cgroup_gib"]) for row in resources], B.PURPLE, 5), ("Host available", [float(row["host_available_gib"]) for row in resources], B.GREEN, 5)],
            0,
            2100,
        ),
        (
            "Physical GPUs 4-7 memory",
            [("Mean GPU memory", [float(row["gpu_mean_gib"]) for row in resources], B.GRPO, 5), ("Max card memory", [float(row["gpu_max_gib"]) for row in resources], B.RED, 5)],
            0,
            80,
        ),
        (
            "Physical GPUs 4-7 utilization (one-minute samples)",
            [
                ("Raw mean", [float(row["gpu_mean_util"]) for row in resources], B.GRPO_LIGHT, 2),
                ("15-min mean", trailing([float(row["gpu_mean_util"]) for row in resources], 15), B.GRPO, 6),
                ("15-min max-card mean", trailing([float(row["gpu_max_util"]) for row in resources], 15), B.RED, 4),
            ],
            0,
            100,
        ),
    ]
    top, height, gap = 145, 500, 25
    for idx, (title, series, low, high) in enumerate(panels):
        B.line_panel(
            draw,
            (45, top + idx * (height + gap), 1395, top + idx * (height + gap) + height),
            elapsed_h,
            series,
            low,
            high,
            title,
            x_label="Hours since observer start",
            x_ticks=hour_ticks,
            x_tick_format=lambda value: f"{value:.1f}" if value == last_h else f"{value:.0f}",
        )

    grpo_steps = [float(row["step"]) for row in aligned]
    grpo_cgroup = [float(row["cgroup_gib"]) for row in aligned]
    ppo_boundary = [row for row in ppo_resource if bool(row["boundary"]) and int(row["step"]) <= 36]
    ppo_map = {float(row["step"]): float(row["cgroup_gib"]) for row in ppo_boundary}
    combined_x = sorted({*grpo_steps, *ppo_map})
    B.line_panel(
        draw,
        (45, 1720, 1395, 2220),
        combined_x,
        [
            ("GRPO nearest-minute", [grpo_cgroup[grpo_steps.index(x)] if x in grpo_steps else None for x in combined_x], B.GRPO, 5),
            ("PPO read-only boundaries", [ppo_map.get(x) for x in combined_x], B.PPO, 6),
        ],
        0,
        2000,
        "Cgroup memory by completed step",
        x_ticks=[1, 5, 10, 15, 20, 25, 30, 35, 36],
    )
    latest = resources[-1]
    draw.rounded_rectangle((45, 2250, 1395, 2425), radius=20, fill=B.WHITE, outline=B.GRID, width=2)
    draw.text((69, 2270), "Latest captured state", fill=B.INK, font=B.F_PANEL)
    cards = [
        ("Cgroup", f"{latest['cgroup_gib']:.1f} GiB"),
        ("Host available", f"{latest['host_available_gib']:.1f} GiB"),
        ("GPU mean", f"{latest['gpu_mean_gib']:.1f} GiB/card"),
        ("GPU util mean", f"{latest['gpu_mean_util']:.1f}%"),
    ]
    for idx, (label, value) in enumerate(cards):
        x = 80 + idx * 323
        draw.text((x, 2332), label, fill=B.GRAY, font=B.F_SMALL)
        draw.text((x, 2362), value, fill=B.INK, font=B.F_CARD)
    img.save(RESOURCE_OUT, optimize=True)


def mean_window(rows: list[dict], start: int, end: int) -> float:
    return statistics.mean(float(row["train_success"]) for row in rows if start <= int(row["step"]) <= end)


def linear_slope_per_step(rows: list[dict], start: int, end: int, key: str) -> float:
    selected = [(float(row["step"]), float(row[key])) for row in rows if start <= int(row["step"]) <= end]
    xbar = statistics.mean(x for x, _ in selected)
    ybar = statistics.mean(y for _, y in selected)
    denominator = sum((x - xbar) ** 2 for x, _ in selected)
    return sum((x - xbar) * (y - ybar) for x, y in selected) / denominator


def summarize(grpo: list[dict], ppo: list[dict], resources: list[dict], aligned: list[dict], ppo_resource: list[dict]) -> dict:
    latest_step = int(grpo[-1]["step"])
    grpo_success = [float(row["train_success"]) for row in grpo]
    ppo_success = [float(row["train_success"]) for row in ppo]
    latest_resource = resources[-1]
    ppo_boundary_map = {int(row["step"]): row for row in ppo_resource if bool(row["boundary"])}
    aligned_map = {int(row["step"]): row for row in aligned}
    common_boundaries = sorted(set(ppo_boundary_map) & set(aligned_map))
    comparisons = {
        str(step): {
            "grpo_cgroup_gib": aligned_map[step]["cgroup_gib"],
            "ppo_cgroup_gib": round(float(ppo_boundary_map[step]["cgroup_gib"]), 3),
            "grpo_minus_ppo_gib": round(float(aligned_map[step]["cgroup_gib"]) - float(ppo_boundary_map[step]["cgroup_gib"]), 3),
        }
        for step in common_boundaries
    }
    gpu_mean_utils = [float(row["gpu_mean_util"]) for row in resources]
    gpu_mean_utils_sorted = sorted(gpu_mean_utils)
    p90_index = math.ceil(0.90 * len(gpu_mean_utils_sorted)) - 1
    return {
        "capture": {
            "complete_steps": latest_step,
            "resource_first_utc": resources[0]["timestamp"].isoformat(),
            "resource_last_utc": resources[-1]["timestamp"].isoformat(),
            "resource_rows": len(resources),
        },
        "success": {
            "grpo": {
                "first": grpo_success[0],
                "latest": grpo_success[-1],
                "mean_all": statistics.mean(grpo_success),
                "trailing5_latest": statistics.mean(grpo_success[-5:]),
                "latest_10_mean": statistics.mean(grpo_success[-10:]),
                "block_1_10": mean_window(grpo, 1, 10),
                "block_11_20": mean_window(grpo, 11, 20),
                "block_21_30": mean_window(grpo, 21, 30),
                "block_31_latest": mean_window(grpo, 31, latest_step),
                "eval": {str(row["step"]): row["eval_success"] for row in grpo if row["eval_success"] is not None},
            },
            "ppo_matched_steps": {
                "latest": ppo_success[-1],
                "mean_all": statistics.mean(ppo_success),
                "trailing5_latest": statistics.mean(ppo_success[-5:]),
                "latest_10_mean": statistics.mean(ppo_success[-10:]),
                "eval": {str(row["step"]): row["eval_success"] for row in ppo if row["eval_success"] is not None},
            },
            "grpo_minus_ppo": {
                "mean_all": statistics.mean(grpo_success) - statistics.mean(ppo_success),
                "trailing5_latest": statistics.mean(grpo_success[-5:]) - statistics.mean(ppo_success[-5:]),
                "latest_10_mean": statistics.mean(grpo_success[-10:]) - statistics.mean(ppo_success[-10:]),
            },
        },
        "optimization": {
            "grpo_latest": {key: grpo[-1][key] for key in ("approx_kl", "clip_fraction", "grad_norm", "ratio_abs", "policy_loss_abs")},
            "grpo_ranges": {key: [min(float(row[key]) for row in grpo), max(float(row[key]) for row in grpo)] for key in ("approx_kl", "clip_fraction", "grad_norm", "ratio_abs")},
            "ppo_latest_matched_step": {key: ppo[-1][key] for key in ("approx_kl", "clip_fraction", "grad_norm")},
        },
        "timing": {
            "elapsed_s": grpo[-1]["elapsed_s"],
            "median_step_s": statistics.median(float(row["step_time_s"]) for row in grpo),
            "median_rollout_s": statistics.median(float(row["generate_rollouts_s"]) for row in grpo),
            "median_actor_s": statistics.median(float(row["actor_training_s"]) for row in grpo),
            "latest10_mean_step_s": statistics.mean(float(row["step_time_s"]) for row in grpo[-10:]),
            "ppo_matched_median_step_s": statistics.median(float(row["step_time_s"]) for row in ppo),
            "ppo_matched_median_rollout_s": statistics.median(float(row["generate_rollouts_s"]) for row in ppo),
            "ppo_matched_median_actor_s": statistics.median(float(row["actor_training_s"]) for row in ppo),
        },
        "resources": {
            "latest": {key: latest_resource[key] for key in ("cgroup_gib", "host_available_gib", "gpu_mean_gib", "gpu_max_gib", "gpu_mean_util", "gpu_max_util")},
            "peak": {
                "cgroup_gib": max(float(row["cgroup_gib"]) for row in resources),
                "gpu_max_gib": max(float(row["gpu_max_gib"]) for row in resources),
            },
            "gpu_utilization_one_minute_samples": {
                "mean_of_four_cards_overall": statistics.mean(gpu_mean_utils),
                "mean_of_four_cards_median": statistics.median(gpu_mean_utils),
                "mean_of_four_cards_p90": gpu_mean_utils_sorted[p90_index],
                "mean_of_four_cards_latest15": statistics.mean(gpu_mean_utils[-15:]),
                "fraction_with_any_card_ge80pct": statistics.mean(float(row["gpu_max_util"]) >= 80.0 for row in resources),
            },
            "aligned": {str(step): aligned_map[step] for step in (1, 10, 20, 30, latest_step)},
            "cgroup_slope_gib_per_step_21_30": linear_slope_per_step(aligned, 21, 30, "cgroup_gib"),
            "cgroup_slope_gib_per_step_31_latest": linear_slope_per_step(aligned, 31, latest_step, "cgroup_gib"),
            "grpo_vs_ppo_at_common_boundaries": comparisons,
        },
        "outputs": [str(path) for path in (SCALARS_OUT, RESOURCE_STEP_OUT, SUCCESS_OUT, OPT_OUT, RESOURCE_OUT)],
    }


def main() -> None:
    grpo = B.parse_grpo(SNAPSHOT / "metrics.log")
    if [int(row["step"]) for row in grpo] != list(range(1, 37)):
        raise ValueError(f"snapshot expected consecutive Step 1-36, got {[row['step'] for row in grpo]}")
    ppo = B.read_ppo(PPO_CSV, len(grpo))
    resources = read_resources(SNAPSHOT / "resource.csv")
    aligned = align_resources(grpo, resources)
    ppo_resource = B.read_ppo_resource(PPO_RESOURCE_CSV)
    B.write_rows(SCALARS_OUT, grpo)
    B.write_rows(RESOURCE_STEP_OUT, aligned)
    render_success(grpo, ppo)
    render_optimization(grpo, ppo)
    render_resources(resources, aligned, ppo_resource)
    summary = summarize(grpo, ppo, resources, aligned, ppo_resource)
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
