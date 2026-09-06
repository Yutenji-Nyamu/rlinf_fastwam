from __future__ import annotations

import csv
import importlib.util
import json
import math
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PIL import Image


ROOT = Path(r"C:\Users\86136\Documents\rl")
EVIDENCE = ROOT / "docs" / "rlinf-shenzhen-pi0-ppo-rlt" / "evidence"
SNAPSHOT = EVIDENCE / "grpo_v2_final_step52_20260823"
BASE_SCRIPT = ROOT / "local_scripts" / "render_shenzhen_grpo_vs_ppo_20260823.py"
PPO_CSV = EVIDENCE / "ppo_formal100_console_scalars_step1_46_20260822.csv"

SCALARS_OUT = SNAPSHOT / "grpo_console_scalars_step1_52.csv"
RESOURCE_STEP_OUT = SNAPSHOT / "grpo_resource_nearest_minute_step1_52.csv"
SUCCESS_OUT = SNAPSHOT / "grpo_vs_ppo_success_step1_52.png"
OPT_OUT = SNAPSHOT / "grpo_vs_ppo_optimization_timing_step1_52.png"
RESOURCE_OUT = SNAPSHOT / "grpo_resource_timeline_through_exit.png"
SUMMARY_OUT = SNAPSHOT / "summary_step52.json"

RAY_THRESHOLD_BYTES = 2_055_933_788_160
RAY_THRESHOLD_GIB = RAY_THRESHOLD_BYTES / 1024**3


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
        statistics.mean(values[index - window + 1 : index + 1]) if index >= window - 1 else None
        for index in range(len(values))
    ]


def pad(values: list[float | None], target: int) -> list[float | None]:
    return values + [None] * (target - len(values))


def trailing_optional(values: list[float | None], window: int = 15) -> list[float | None]:
    output: list[float | None] = []
    for index in range(len(values)):
        if index < window - 1:
            output.append(None)
            continue
        selected = [value for value in values[index - window + 1 : index + 1] if value is not None]
        output.append(statistics.mean(selected) if selected else None)
    return output


def eval_steps(rows: list[dict]) -> set[int]:
    return {int(row["step"]) for row in rows if row["eval_success"] is not None}


def render_success(grpo: list[dict], ppo: list[dict]) -> None:
    steps = [float(row["step"]) for row in grpo]
    grpo_success = [float(row["train_success"]) for row in grpo]
    ppo_success = [float(row["train_success"]) for row in ppo]
    ppo_raw = pad(ppo_success, len(grpo))
    ppo_trailing = pad(trailing(ppo_success), len(grpo))
    shaded = eval_steps(grpo) | eval_steps(ppo)

    image = Image.new("RGB", (1440, 1360), B.LIGHT)
    draw = B.add_header(
        image,
        "GRPO through Step 52 vs PPO through Step 46",
        "Matched 4-GPU / 512-trajectory outer-step budget; PPO evidence ends at Step 46",
    )
    B.line_panel(
        draw,
        (45, 145, 1395, 705),
        steps,
        [("GRPO raw", grpo_success, B.GRPO_LIGHT, 3), ("PPO raw", ppo_raw, B.PPO_LIGHT, 3)],
        0.65,
        1.0,
        "Per-step training-rollout success",
        eval_steps=shaded,
    )
    plot = B.line_panel(
        draw,
        (45, 735, 1395, 1295),
        steps,
        [("GRPO trailing-5", trailing(grpo_success), B.GRPO, 7), ("PPO trailing-5", ppo_trailing, B.PPO, 7)],
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
    image.save(SUCCESS_OUT, optimize=True)


def render_optimization(grpo: list[dict], ppo: list[dict]) -> None:
    steps = [float(row["step"]) for row in grpo]
    shaded = eval_steps(grpo) | eval_steps(ppo)
    panels = [
        ("Approximate KL", "approx_kl", 0.0, 0.08),
        ("Clip fraction", "clip_fraction", 0.0, 0.14),
        ("Pre-clip gradient norm", "grad_norm", 0.0, 45.0),
        ("Whole-step wall time", "step_time_s", 0.0, 2100.0),
    ]
    image = Image.new("RGB", (1440, 1950), B.LIGHT)
    draw = B.add_header(
        image,
        "GRPO optimization and time through Step 52",
        "PPO comparison is available through Step 46; GRPO Step 53 never completed an optimizer update",
    )
    top, height, gap = 145, 410, 28
    for index, (title, key, low, high) in enumerate(panels):
        ppo_values = pad([float(row[key]) for row in ppo], len(grpo))
        B.line_panel(
            draw,
            (45, top + index * (height + gap), 1395, top + index * (height + gap) + height),
            steps,
            [("GRPO", [float(row[key]) for row in grpo], B.GRPO, 5), ("PPO", ppo_values, B.PPO, 5)],
            low,
            high,
            title,
            eval_steps=shaded,
        )
    image.save(OPT_OUT, optimize=True)


def optional_float(raw: dict[str, str], key: str) -> float | None:
    value = raw.get(key, "")
    return float(value) if value else None


def read_resources(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            gpu_mem = [optional_float(raw, f"gpu{gpu}_used_mib") for gpu in range(4, 8)]
            gpu_util = [optional_float(raw, f"gpu{gpu}_util_pct") for gpu in range(4, 8)]
            present_mem = [value / 1024 for value in gpu_mem if value is not None]
            present_util = [value for value in gpu_util if value is not None]
            cgroup_bytes = optional_float(raw, "cgroup_memory_current_bytes")
            rows.append(
                {
                    "timestamp": datetime.fromisoformat(raw["timestamp"]),
                    "driver_alive": int(raw["driver_alive"]),
                    "cgroup_gib": cgroup_bytes / 1024**3 if cgroup_bytes is not None else None,
                    "host_available_gib": float(raw["host_mem_available_kib"]) / 1024**2,
                    "gpu_mean_gib": statistics.mean(present_mem) if present_mem else None,
                    "gpu_max_gib": max(present_mem) if present_mem else None,
                    "gpu_mean_util": statistics.mean(present_util) if present_util else None,
                    "gpu_max_util": max(present_util) if present_util else None,
                }
            )
    if len(rows) < 2:
        raise ValueError("resource.csv has fewer than two samples")
    return rows


def align_resources(grpo: list[dict], resources: list[dict]) -> list[dict]:
    start = resources[0]["timestamp"]
    complete = [row for row in resources if row["cgroup_gib"] is not None and row["gpu_mean_gib"] is not None]
    output: list[dict] = []
    for metric in grpo:
        target = start + timedelta(seconds=float(metric["elapsed_s"]))
        nearest = min(complete, key=lambda row: abs((row["timestamp"] - target).total_seconds()))
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


def render_resources(resources: list[dict]) -> None:
    start = resources[0]["timestamp"]
    elapsed_h = [(row["timestamp"] - start).total_seconds() / 3600 for row in resources]
    last_h = elapsed_h[-1]
    hour_ticks = [value for value in range(0, math.ceil(last_h) + 1, 2) if abs(last_h - value) > 0.75]
    hour_ticks.append(last_h)
    gpu_mean = [row["gpu_mean_gib"] for row in resources]
    gpu_max = [row["gpu_max_gib"] for row in resources]
    util_mean = [row["gpu_mean_util"] for row in resources]
    util_max = [row["gpu_max_util"] for row in resources]

    image = Image.new("RGB", (1440, 1900), B.LIGHT)
    exit_cst = resources[-1]["timestamp"].astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M CST")
    draw = B.add_header(
        image,
        "GRPO resource timeline through Ray memory exit",
        f"One-minute observer samples through {exit_cst}; Ray killed workers at its 95% node-memory threshold",
    )
    panels = [
        (
            "Main memory and Ray 95% threshold",
            [
                ("GRPO cgroup used", [row["cgroup_gib"] for row in resources], B.PURPLE, 5),
                ("Host available", [row["host_available_gib"] for row in resources], B.GREEN, 5),
                ("Ray threshold", [RAY_THRESHOLD_GIB] * len(resources), B.RED, 3),
            ],
            0,
            2050,
        ),
        (
            "Physical GPUs 4-7 memory",
            [("Mean GPU memory", gpu_mean, B.GRPO, 5), ("Max card memory", gpu_max, B.RED, 5)],
            0,
            80,
        ),
        (
            "Physical GPUs 4-7 utilization (one-minute samples)",
            [
                ("Raw mean", util_mean, B.GRPO_LIGHT, 2),
                ("15-min mean", trailing_optional(util_mean), B.GRPO, 6),
                ("15-min max-card mean", trailing_optional(util_max), B.RED, 4),
            ],
            0,
            100,
        ),
    ]
    top, height, gap = 145, 500, 25
    for index, (title, series, low, high) in enumerate(panels):
        B.line_panel(
            draw,
            (45, top + index * (height + gap), 1395, top + index * (height + gap) + height),
            elapsed_h,
            series,
            low,
            high,
            title,
            x_label="Hours since observer start",
            x_ticks=hour_ticks,
            x_tick_format=lambda value: f"{value:.1f}" if value == last_h else f"{value:.0f}",
        )

    pre_exit = next(row for row in reversed(resources) if row["cgroup_gib"] is not None)
    draw.rounded_rectangle((45, 1720, 1395, 1840), radius=20, fill=B.WHITE, outline=B.GRID, width=2)
    cards = [
        ("Pre-exit cgroup", f"{pre_exit['cgroup_gib']:.1f} GiB"),
        ("Host available", f"{pre_exit['host_available_gib']:.1f} GiB"),
        ("Max card memory", f"{pre_exit['gpu_max_gib']:.1f} GiB"),
        ("Driver exit", "Ray memory monitor"),
    ]
    for index, (label, value) in enumerate(cards):
        x = 80 + index * 323
        draw.text((x, 1743), label, fill=B.GRAY, font=B.F_SMALL)
        draw.text((x, 1780), value, fill=B.INK, font=B.F_CARD)
    image.save(RESOURCE_OUT, optimize=True)


def write_resource_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(grpo: list[dict], ppo: list[dict], resources: list[dict], aligned: list[dict]) -> dict:
    grpo_success = [float(row["train_success"]) for row in grpo]
    ppo_success = [float(row["train_success"]) for row in ppo]
    common_grpo = grpo[: len(ppo)]
    common_success = [float(row["train_success"]) for row in common_grpo]
    gpu_mem = [float(row["gpu_max_gib"]) for row in resources if row["gpu_max_gib"] is not None]
    cgroup = [float(row["cgroup_gib"]) for row in resources if row["cgroup_gib"] is not None]
    pre_exit = next(row for row in reversed(resources) if row["cgroup_gib"] is not None)
    return {
        "capture": {
            "complete_steps": len(grpo),
            "incomplete_next_step": 53,
            "resource_rows": len(resources),
            "resource_first_utc": resources[0]["timestamp"].isoformat(),
            "resource_last_utc": resources[-1]["timestamp"].isoformat(),
        },
        "success": {
            "grpo_first": grpo_success[0],
            "grpo_latest": grpo_success[-1],
            "grpo_mean_all": statistics.mean(grpo_success),
            "grpo_trailing5": statistics.mean(grpo_success[-5:]),
            "grpo_latest10": statistics.mean(grpo_success[-10:]),
            "grpo_fixed64": {str(row["step"]): row["eval_success"] for row in grpo if row["eval_success"] is not None},
            "common_step_1_46": {
                "grpo_mean": statistics.mean(common_success),
                "ppo_mean": statistics.mean(ppo_success),
                "grpo_minus_ppo": statistics.mean(common_success) - statistics.mean(ppo_success),
                "grpo_trailing5": statistics.mean(common_success[-5:]),
                "ppo_trailing5": statistics.mean(ppo_success[-5:]),
            },
        },
        "optimization": {
            "latest": {key: grpo[-1][key] for key in ("approx_kl", "clip_fraction", "grad_norm", "ratio_abs", "policy_loss_abs")},
            "ranges": {key: [min(float(row[key]) for row in grpo), max(float(row[key]) for row in grpo)] for key in ("approx_kl", "clip_fraction", "grad_norm", "ratio_abs")},
        },
        "timing": {
            "elapsed_s_through_step52": grpo[-1]["elapsed_s"],
            "median_step_s": statistics.median(float(row["step_time_s"]) for row in grpo),
            "median_rollout_s": statistics.median(float(row["generate_rollouts_s"]) for row in grpo),
            "median_actor_s": statistics.median(float(row["actor_training_s"]) for row in grpo),
            "latest10_mean_step_s": statistics.mean(float(row["step_time_s"]) for row in grpo[-10:]),
        },
        "resources": {
            "peak_cgroup_gib": max(cgroup),
            "minimum_host_available_gib": min(float(row["host_available_gib"]) for row in resources),
            "peak_gpu_card_gib": max(gpu_mem),
            "pre_exit": {key: pre_exit[key] for key in ("timestamp", "cgroup_gib", "host_available_gib", "gpu_mean_gib", "gpu_max_gib")},
            "post_exit_host_available_gib": resources[-1]["host_available_gib"],
            "aligned_step52": aligned[-1],
        },
        "exit": {
            "exit_code": 255,
            "exit_time_utc": "2026-08-23T12:01:56+00:00",
            "cause": "Ray userspace memory monitor killed four workers after node usage crossed its 95% threshold",
            "ray_reported_usage_gb": 1914.32,
            "ray_reported_total_gb": 2015.51,
            "ray_reported_fraction": 0.949793,
            "threshold_bytes": RAY_THRESHOLD_BYTES,
            "largest_workers_gb": [528.25, 438.57, 430.64, 406.83],
            "kernel_or_cgroup_oom": False,
        },
        "outputs": [str(path) for path in (SCALARS_OUT, RESOURCE_STEP_OUT, SUCCESS_OUT, OPT_OUT, RESOURCE_OUT)],
    }


def main() -> None:
    grpo = B.parse_grpo(SNAPSHOT / "metrics.log")
    if [int(row["step"]) for row in grpo] != list(range(1, 53)):
        raise ValueError("expected consecutive complete GRPO Step 1-52")
    ppo = B.read_ppo(PPO_CSV, 46)
    resources = read_resources(SNAPSHOT / "resource.csv")
    aligned = align_resources(grpo, resources)

    B.write_rows(SCALARS_OUT, grpo)
    write_resource_rows(RESOURCE_STEP_OUT, aligned)
    render_success(grpo, ppo)
    render_optimization(grpo, ppo)
    render_resources(resources)
    summary = summarize(grpo, ppo, resources, aligned)
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
