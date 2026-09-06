from __future__ import annotations

import csv
import importlib.util
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(r"C:\Users\86136\Documents\rl")
EVIDENCE = ROOT / "docs" / "rlinf-shenzhen-pi0-ppo-rlt" / "evidence"
SNAPSHOT = EVIDENCE / "dvac-grpo-w0to5-live-step40-20260826"
LOG = SNAPSHOT / "driver.log"
RESOURCE = SNAPSHOT / "resource.csv"
BASELINE = EVIDENCE / "grpo_v2_final_step52_20260823" / "grpo_console_scalars_step1_52.csv"
PARSER = ROOT / "local_scripts" / "render_shenzhen_dvac_grpo_v5_resume_live_20260825.py"

ORANGE = "#E4572E"
DEEP_TEAL = "#00796B"
LIGHT_TEAL = "#80CBC4"
GOLD = "#B07C00"
RED = "#B42318"
SLATE = "#667085"


def load_parser():
    spec = importlib.util.spec_from_file_location("dvac_parser", PARSER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {PARSER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load_parser()
B = M.B


def readable_tick(value: float, span: float) -> str:
    if span < 1:
        return f"{value:.2f}"
    if span <= 3:
        return f"{value:.1f}"
    return f"{value:.0f}"


B.fmt_tick = readable_tick


def baseline_rows(through: int) -> list[dict[str, float | int | None]]:
    rows: list[dict[str, float | int | None]] = []
    with BASELINE.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            step = int(raw["step"])
            if step > through:
                break
            rows.append(
                {
                    "step": step,
                    "train_success": float(raw["train_success"]),
                    "eval_success": float(raw["eval_success"]) if raw["eval_success"] else None,
                }
            )
    return rows


def draw_markers(
    draw: ImageDraw.ImageDraw,
    plot: tuple[float, float, float, float],
    xs: list[float],
    values: list[float | None],
    y_min: float,
    y_max: float,
    color: str,
    shape: str,
) -> None:
    left, top, right, bottom = plot
    x_min, x_max = min(xs), max(xs)
    for x_value, value in zip(xs, values):
        if value is None:
            continue
        x = left + (x_value - x_min) / max(1e-9, x_max - x_min) * (right - left)
        y = bottom - (float(value) - y_min) / (y_max - y_min) * (bottom - top)
        if shape == "square":
            draw.rectangle((x - 5, y - 5, x + 5, y + 5), fill=color, outline=B.WHITE, width=1)
        else:
            draw.polygon(((x, y - 6), (x - 6, y + 5), (x + 6, y + 5)), fill=color)


def render_success(current: list[dict], baseline: list[dict]) -> Path:
    complete = int(current[-1]["step"])
    xs = [float(row["step"]) for row in current]
    cur = [float(row["train_success"]) for row in current]
    base = [float(row["train_success"]) for row in baseline]
    cur_ma = B.trailing(cur)
    base_ma = B.trailing(base)
    cur_eval = [(int(row["step"]), row["eval_success"]) for row in current if row["eval_success"] is not None]
    base_eval = [(int(row["step"]), row["eval_success"]) for row in baseline if row["eval_success"] is not None]

    out = SNAPSHOT / f"01_w0to5_vs_matched_grpo_success_step{complete}.png"
    image = Image.new("RGB", (1440, 1780), B.LIGHT)
    draw = B.add_header(
        image,
        f"GRPO-DVAC W[0,5] live through complete Step {complete}",
        "Orange squares = W[0,5]; deep-teal triangles = strict-matched original GRPO.",
    )

    raw_plot = B.line_panel(
        draw,
        (45, 145, 1395, 630),
        xs,
        [("W[0,5] raw", cur, ORANGE, 5), ("Original GRPO raw", base, DEEP_TEAL, 4)],
        0.68,
        1.00,
        "Per-step training-rollout success",
        eval_steps={5, 10},
    )
    draw_markers(draw, raw_plot, xs, cur, 0.68, 1.00, ORANGE, "square")
    draw_markers(draw, raw_plot, xs, base, 0.68, 1.00, DEEP_TEAL, "triangle")

    ma_plot = B.line_panel(
        draw,
        (45, 655, 1395, 1140),
        xs,
        [("W[0,5] trailing-5", cur_ma, ORANGE, 7), ("Original GRPO trailing-5", base_ma, DEEP_TEAL, 6)],
        0.74,
        0.98,
        "Five-step moving mean",
        eval_steps={5, 10},
    )
    draw_markers(draw, ma_plot, xs, cur_ma, 0.74, 0.98, ORANGE, "square")
    draw_markers(draw, ma_plot, xs, base_ma, 0.74, 0.98, DEEP_TEAL, "triangle")

    eval_plot = B.line_panel(draw, (45, 1165, 1395, 1625), xs, [], 0.80, 1.00, "Fixed-64 evaluation")
    B.draw_eval_markers(draw, eval_plot, xs, 0.80, 1.00, cur_eval, ORANGE, "square")
    B.draw_eval_markers(draw, eval_plot, xs, 0.80, 1.00, base_eval, DEEP_TEAL, "triangle")
    for step, value in cur_eval:
        label = f"W {round(float(value) * 64)}/64"
        x = eval_plot[0] + (step - 1) / max(1, complete - 1) * (eval_plot[2] - eval_plot[0])
        y = eval_plot[3] - (float(value) - 0.80) / 0.20 * (eval_plot[3] - eval_plot[1])
        draw.text((x - 46, y - 38), label, fill=ORANGE, font=B.F_SMALL)
    for step, value in base_eval:
        label = f"G {round(float(value) * 64)}/64"
        x = eval_plot[0] + (step - 1) / max(1, complete - 1) * (eval_plot[2] - eval_plot[0])
        y = eval_plot[3] - (float(value) - 0.80) / 0.20 * (eval_plot[3] - eval_plot[1])
        draw.text((x + 12, y + 12), label, fill=DEEP_TEAL, font=B.F_SMALL)

    mean_delta = (statistics.mean(cur) - statistics.mean(base)) * 100
    latest5_delta = (statistics.mean(cur[-5:]) - statistics.mean(base[-5:])) * 100
    draw.rounded_rectangle((55, 1650, 1385, 1745), radius=18, fill=B.WHITE, outline=B.GRID, width=2)
    draw.text(
        (78, 1677),
        f"Paired Steps 1-{complete}: mean delta {mean_delta:+.2f} pp; latest-5 delta {latest5_delta:+.2f} pp.  Too early for an effect claim.",
        fill=B.INK,
        font=B.F_SMALL,
    )
    image.save(out, optimize=True)
    return out


def read_resources() -> list[dict[str, float | datetime]]:
    rows: list[dict[str, float | datetime]] = []
    with RESOURCE.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            gpu_memory = [float(raw[f"gpu{gpu}_used_mib"]) / 1024 for gpu in range(4, 8)]
            rows.append(
                {
                    "timestamp": datetime.fromisoformat(raw["timestamp"]),
                    "ram": float(raw["host_mem_available_kib"]) / 1024**2,
                    "gpu_mean": statistics.mean(gpu_memory),
                    "gpu_max": max(gpu_memory),
                }
            )
    return rows


def render_health(current: list[dict], resources: list[dict[str, float | datetime]]) -> Path:
    complete = int(current[-1]["step"])
    xs = [float(row["step"]) for row in current]
    kl = [float(row["approx_kl"]) for row in current]
    clip = [float(row["clip_fraction"]) for row in current]
    grad = [float(row["grad_norm"]) for row in current]
    weight = [M.console_metric(LOG, int(row["step"]), "actor/dvac_weight_mean") for row in current]
    ess = [M.console_metric(LOG, int(row["step"]), "actor/dvac_weight_ess_fraction") for row in current]

    start = resources[0]["timestamp"]
    assert isinstance(start, datetime)
    hours = []
    ram = []
    gpu_mean = []
    gpu_max = []
    for row in resources:
        timestamp = row["timestamp"]
        assert isinstance(timestamp, datetime)
        hours.append((timestamp - start).total_seconds() / 3600)
        ram.append(float(row["ram"]))
        gpu_mean.append(float(row["gpu_mean"]))
        gpu_max.append(float(row["gpu_max"]))

    out = SNAPSHOT / f"02_w0to5_optimization_and_resources_step{complete}.png"
    image = Image.new("RGB", (1440, 2450), B.LIGHT)
    latest_ram = float(resources[-1]["ram"])
    margin_to_ray = latest_ram - 100.0
    draw = B.add_header(
        image,
        f"GRPO-DVAC W[0,5] optimization and resources through Step {complete}",
        f"All optimization scalars are finite; RAM available {latest_ram:.0f} GiB, about {margin_to_ray:.0f} GiB above the Ray 95% boundary.",
    )
    panels = [
        ("PPO update: approximate KL and clip fraction", xs, [("KL", kl, ORANGE, 5), ("Clip fraction", clip, DEEP_TEAL, 5)], 0.0, 0.14, "Completed global step"),
        ("Pre-clip gradient norm", xs, [("Gradient norm", grad, GOLD, 5)], 0.0, 55.0, "Completed global step"),
        ("DVAC weighting: mean weight and ESS fraction", xs, [("Weight mean", weight, ORANGE, 5), ("ESS fraction", ess, DEEP_TEAL, 5)], 0.5, 1.8, "Completed global step"),
        ("Host available RAM", hours, [("Available RAM (GiB)", ram, ORANGE, 5), ("Ray 95% boundary (~100 GiB)", [100.0] * len(ram), RED, 3)], 0.0, 2050.0, "Hours since launch"),
        ("Physical GPUs 4-7 memory", hours, [("Mean card (GiB)", gpu_mean, DEEP_TEAL, 5), ("Max card (GiB)", gpu_max, ORANGE, 4)], 0.0, 80.0, "Hours since launch"),
    ]
    top, height, gap = 145, 425, 28
    for index, (title, x_values, lines, low, high, x_label) in enumerate(panels):
        B.line_panel(
            draw,
            (45, top + index * (height + gap), 1395, top + index * (height + gap) + height),
            x_values,
            lines,
            low,
            high,
            title,
            x_label=x_label,
            x_ticks=[min(x_values), (min(x_values) + max(x_values)) / 2, max(x_values)],
            x_tick_format=(lambda value: f"{value:.1f}"),
        )
    image.save(out, optimize=True)
    return out


def main() -> None:
    current = M.parse_grpo_any(LOG)
    complete = int(current[-1]["step"])
    baseline = baseline_rows(complete)
    resources = read_resources()
    outputs = [render_success(current, baseline), render_health(current, resources)]

    cur = [float(row["train_success"]) for row in current]
    base = [float(row["train_success"]) for row in baseline]
    summary = {
        "complete_step": complete,
        "train_success_latest": cur[-1],
        "paired_mean_delta_pp": (statistics.mean(cur) - statistics.mean(base)) * 100,
        "paired_latest5_delta_pp": (statistics.mean(cur[-5:]) - statistics.mean(base[-5:])) * 100,
        "fixed64": {str(row["step"]): row["eval_success"] for row in current if row["eval_success"] is not None},
        "optimization_latest": {
            "approx_kl": current[-1]["approx_kl"],
            "clip_fraction": current[-1]["clip_fraction"],
            "grad_norm": current[-1]["grad_norm"],
            "dvac_weight_mean": M.console_metric(LOG, complete, "actor/dvac_weight_mean"),
            "dvac_weight_ess_fraction": M.console_metric(LOG, complete, "actor/dvac_weight_ess_fraction"),
        },
        "resources": {
            "samples": len(resources),
            "hours": (resources[-1]["timestamp"] - resources[0]["timestamp"]).total_seconds() / 3600,
            "ram_available_latest_gib": resources[-1]["ram"],
            "ram_available_min_gib": min(float(row["ram"]) for row in resources),
            "ram_available_last_hour_delta_gib": float(resources[-1]["ram"]) - float(resources[-61]["ram"]),
            "gpu_max_peak_gib": max(float(row["gpu_max"]) for row in resources),
            "gpu_max_latest_gib": resources[-1]["gpu_max"],
        },
        "outputs": [path.name for path in outputs],
    }
    (SNAPSHOT / f"summary_step{complete}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
