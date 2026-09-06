from __future__ import annotations

import csv
import importlib.util
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

from PIL import Image


ROOT = Path(r"C:\Users\86136\Documents\rl")
EVIDENCE = ROOT / "docs" / "rlinf-shenzhen-pi0-ppo-rlt" / "evidence"
SNAPSHOT = EVIDENCE / "dvac-grpo-v4-live-step31-20260825"
BASELINE_CSV = EVIDENCE / "grpo_v2_final_step52_20260823" / "grpo_console_scalars_step1_52.csv"
BASE_SCRIPT = ROOT / "local_scripts" / "render_shenzhen_grpo_vs_ppo_20260823.py"


def load_base():
    spec = importlib.util.spec_from_file_location("grpo_plot_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B = load_base()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def tag_values(tb: dict, tag: str) -> list[float]:
    values = tb[tag]
    expected = list(range(len(values)))
    actual = [int(item["step"]) for item in values]
    if actual != expected:
        raise ValueError(f"nonconsecutive TensorBoard steps for {tag}: {actual}")
    return [float(item["value"]) for item in values]


def _vertical_cut(draw, plot, x_min: float, x_max: float, cut: float, label: str) -> None:
    left, top, right, bottom = plot
    x = left + (cut - x_min) / (x_max - x_min) * (right - left)
    for y in range(int(top), int(bottom), 18):
        draw.line((x, y, x, min(y + 9, bottom)), fill=B.GRAY, width=2)
    draw.text((x + 8, top + 48), label, fill=B.GRAY, font=B.F_SMALL)


def _label_eval(draw, plot, x_min: float, x_max: float, y_min: float, y_max: float, step: float, value: float, text: str, color: str, dx: int, dy: int) -> None:
    left, top, right, bottom = plot
    x = left + (step - x_min) / (x_max - x_min) * (right - left)
    y = bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
    draw.text((x + dx, y + dy), text, fill=color, font=B.F_SMALL)


def render_success(current: list[dict], baseline: list[dict], evals: dict[int, float]) -> Path:
    current_steps = [float(row["step"]) for row in current]
    full_steps = [float(row["step"]) for row in baseline]
    complete = int(current_steps[-1])
    cur = [float(row["train_success"]) for row in current]
    base = [float(row["train_success"]) for row in baseline]
    current_padded = cur + [None] * (len(base) - len(cur))
    current_ma = B.trailing(cur) + [None] * (len(base) - len(cur))
    baseline_ma = B.trailing(base)
    baseline_paired = [value if index < complete else None for index, value in enumerate(base)]
    baseline_paired_ma = [value if index < complete else None for index, value in enumerate(baseline_ma)]
    baseline_future = [None if index < complete - 1 else value for index, value in enumerate(base)]
    baseline_future_ma = [None if index < complete - 1 else value for index, value in enumerate(baseline_ma)]
    baseline_evals = [(int(row["step"]), float(row["eval_success"])) for row in baseline if row["eval_success"]]
    out = SNAPSHOT / "01_dvac_v4_vs_grpo_v2_success_step1_31_full_baseline52.png"
    image = Image.new("RGB", (1440, 1840), B.LIGHT)
    draw = B.add_header(
        image,
        "Strict-matched Shenzhen GRPO-DVAC through Step 31",
        "Purple = DVAC; blue = matched GRPO. Baseline continues to Step 52 only as unpaired context.",
    )
    raw_plot = B.line_panel(
        draw,
        (45, 145, 1395, 630),
        full_steps,
        [
            ("DVAC raw", current_padded, B.PURPLE, 4),
            ("GRPO raw (paired)", baseline_paired, B.GRPO, 4),
            ("GRPO baseline-only", baseline_future, B.GRPO_LIGHT, 3),
        ],
        0.65,
        1.0,
        "Per-step training-rollout success",
    )
    ma_plot = B.line_panel(
        draw,
        (45, 655, 1395, 1140),
        full_steps,
        [
            ("DVAC 5-step mean", current_ma, B.PURPLE, 7),
            ("GRPO 5-step mean (paired)", baseline_paired_ma, B.GRPO, 7),
            ("GRPO baseline-only", baseline_future_ma, B.GRPO_LIGHT, 4),
        ],
        0.65,
        1.0,
        "Five-step moving mean",
    )
    eval_plot = B.line_panel(
        draw,
        (45, 1165, 1395, 1685),
        full_steps,
        [],
        0.84,
        1.0,
        "Fixed-64 evaluation",
    )
    for plot in (raw_plot, ma_plot, eval_plot):
        _vertical_cut(draw, plot, 1.0, 52.0, complete + 0.5, f"after Step {complete}: baseline only")
    current_eval_markers = [(step - 0.25, value) for step, value in evals.items()]
    baseline_eval_markers = [(step + 0.25, value) for step, value in baseline_evals]
    B.draw_eval_markers(draw, eval_plot, full_steps, 0.84, 1.0, current_eval_markers, B.PURPLE, "square")
    B.draw_eval_markers(
        draw,
        eval_plot,
        full_steps,
        0.84,
        1.0,
        baseline_eval_markers,
        B.GRPO,
        "triangle",
    )
    legend_y = eval_plot[1] + 14
    draw.rectangle((eval_plot[0] + 12, legend_y, eval_plot[0] + 30, legend_y + 18), fill=B.PURPLE)
    draw.text((eval_plot[0] + 40, legend_y - 2), "purple square = DVAC", fill=B.PURPLE, font=B.F_SMALL)
    lx = eval_plot[0] + 255
    draw.polygon(((lx + 9, legend_y), (lx, legend_y + 18), (lx + 18, legend_y + 18)), fill=B.GRPO)
    draw.text((lx + 28, legend_y - 2), "blue triangle = matched GRPO", fill=B.GRPO, font=B.F_SMALL)
    baseline_eval_map = dict(baseline_evals)
    for step, value in sorted(evals.items()):
        base_value = baseline_eval_map[step]
        label = f"D {round(value * 64):.0f} | G {round(base_value * 64):.0f}"
        _label_eval(draw, eval_plot, 1.0, 52.0, 0.84, 1.0, step, max(value, base_value), label, B.INK, -50, -32)
    for step, value in baseline_evals:
        if step > complete:
            _label_eval(draw, eval_plot, 1.0, 52.0, 0.84, 1.0, step + 0.25, value, f"G {round(value * 64):.0f}/64", B.GRPO, 12, 8)
    paired_base = base[:complete]
    fixed_dvac = sum(round(value * 64) for value in evals.values())
    fixed_base = sum(round(value * 64) for step, value in baseline_evals if step <= complete)
    card = (
        f"Paired Step 1-{complete}: train mean DVAC-GRPO = "
        f"{(statistics.mean(cur) - statistics.mean(paired_base)) * 100:+.2f} pp; "
        f"latest-5 = {(statistics.mean(cur[-5:]) - statistics.mean(paired_base[-5:])) * 100:+.2f} pp.  "
        f"Step 30 fixed = 63/64 vs 62/64 (+1/64); cumulative = {fixed_dvac}/192 vs {fixed_base}/192."
    )
    draw.rounded_rectangle((55, 1710, 1385, 1805), radius=18, fill=B.WHITE, outline=B.GRID, width=2)
    draw.text((78, 1738), card, fill=B.INK, font=B.F_SMALL)
    image.save(out, optimize=True)
    return out


def render_optimization(current: list[dict], tb: dict) -> Path:
    steps = [float(row["step"]) for row in current]
    kl = [float(row["approx_kl"]) for row in current]
    clip = [float(row["clip_fraction"]) for row in current]
    grad = [float(row["grad_norm"]) for row in current]
    mean = tag_values(tb, "train/actor/dvac_weight_mean")
    sq_mean = tag_values(tb, "train/actor/dvac_weight_sq_mean")
    std = [math.sqrt(max(0.0, sq - avg * avg)) for avg, sq in zip(mean, sq_mean)]
    ess = tag_values(tb, "train/actor/dvac_weight_ess_fraction")
    low_clip = tag_values(tb, "train/actor/dvac_z_low_clip_fraction")
    high_clip = tag_values(tb, "train/actor/dvac_z_high_clip_fraction")
    if not all(len(values) == len(current) for values in (mean, std, ess, low_clip, high_clip)):
        raise ValueError("TensorBoard/current row count mismatch")

    out = SNAPSHOT / "02_dvac_v4_optimization_and_weights_step1_31.png"
    image = Image.new("RGB", (1440, 2040), B.LIGHT)
    draw = B.add_header(
        image,
        "GRPO-DVAC optimization and weighting through Step 31",
        "All optimization scalars are finite; Step 1 is the intentional uniform-weight warm-up",
    )
    panels = [
        ("Approximate KL and PPO clip fraction", [("KL", kl, B.PURPLE, 5), ("Clip fraction", clip, B.GRPO, 5)], 0.0, 0.13),
        ("Pre-clip gradient norm", [("Gradient norm", grad, B.GREEN, 5)], 0.0, 40.0),
        ("DVAC weight balance", [("Weight mean", mean, B.PURPLE, 5), ("ESS fraction", ess, B.GREEN, 5), ("Weight std", std, B.GRPO, 5)], 0.0, 1.10),
        ("DVAC z clipping", [("Upper clip fraction", high_clip, B.RED, 5), ("Lower clip fraction", low_clip, B.PPO, 5)], 0.0, 0.06),
    ]
    top, height, gap = 145, 430, 25
    for index, (title, series, low, high) in enumerate(panels):
        B.line_panel(
            draw,
            (45, top + index * (height + gap), 1395, top + index * (height + gap) + height),
            steps,
            series,
            low,
            high,
            title,
        )
    image.save(out, optimize=True)
    return out


def moving_optional(values: list[float], window: int) -> list[float | None]:
    return [statistics.mean(values[index - window + 1 : index + 1]) if index >= window - 1 else None for index in range(len(values))]


def read_resources(path: Path) -> list[dict]:
    rows = []
    for raw in read_csv(path):
        mem = [float(raw[f"gpu{gpu}_used_mib"]) / 1024 for gpu in range(4, 8)]
        util = [float(raw[f"gpu{gpu}_util_pct"]) for gpu in range(4, 8)]
        rows.append(
            {
                "timestamp": datetime.fromisoformat(raw["timestamp"]),
                "host_available_gib": float(raw["host_mem_available_kib"]) / 1024**2,
                "gpu_mean_gib": statistics.mean(mem),
                "gpu_max_gib": max(mem),
                "gpu_mean_util": statistics.mean(util),
                "gpu_max_util": max(util),
            }
        )
    return rows


def render_resources(resources: list[dict]) -> Path:
    start = resources[0]["timestamp"]
    hours = [(row["timestamp"] - start).total_seconds() / 3600 for row in resources]
    host = [float(row["host_available_gib"]) for row in resources]
    gpu_mean = [float(row["gpu_mean_gib"]) for row in resources]
    gpu_max = [float(row["gpu_max_gib"]) for row in resources]
    util = [float(row["gpu_mean_util"]) for row in resources]
    last = hours[-1]
    ticks = [float(value) for value in range(0, math.floor(last) + 1, 2)]
    if not ticks or abs(ticks[-1] - last) > 0.5:
        ticks.append(last)

    out = SNAPSHOT / "03_dvac_v4_resource_timeline_through_step31.png"
    image = Image.new("RGB", (1440, 1790), B.LIGHT)
    draw = B.add_header(
        image,
        "GRPO-DVAC resource timeline through Step 31",
        "One-minute observer samples; the 100 GiB line approximates Ray's 95% whole-node memory boundary",
    )
    panels = [
        ("Host available memory", [("Available RAM", host, B.GREEN, 5), ("Approx. Ray boundary", [100.0] * len(host), B.RED, 3)], 0.0, 2050.0),
        ("Physical GPUs 4-7 memory", [("Mean card", gpu_mean, B.GRPO, 5), ("Max card", gpu_max, B.RED, 5)], 0.0, 80.0),
        ("Physical GPUs 4-7 utilization", [("Raw mean", util, B.GRPO_LIGHT, 2), ("15-minute mean", moving_optional(util, 15), B.GRPO, 6)], 0.0, 100.0),
    ]
    top, height, gap = 145, 460, 25
    for index, (title, series, low, high) in enumerate(panels):
        B.line_panel(
            draw,
            (45, top + index * (height + gap), 1395, top + index * (height + gap) + height),
            hours,
            series,
            low,
            high,
            title,
            x_label="Hours since launch",
            x_ticks=ticks,
            x_tick_format=lambda value: f"{value:.1f}" if abs(value - last) < 0.01 else f"{value:.0f}",
        )
    image.save(out, optimize=True)
    return out


def main() -> None:
    current = B.parse_grpo(SNAPSHOT / "driver.log")
    if [int(row["step"]) for row in current] != list(range(1, 32)):
        raise ValueError("expected complete Step 1-31")
    baseline = read_csv(BASELINE_CSV)
    tb = json.loads((SNAPSHOT / "tensorboard_scalars.json").read_text(encoding="utf-8"))
    resources = read_resources(SNAPSHOT / "resource.csv")
    evals = {int(item["step"]) + 1: float(item["value"]) for item in tb["eval/success_once"]}

    current_success = [float(row["train_success"]) for row in current]
    baseline_success = [float(row["train_success"]) for row in baseline[: len(current)]]
    mean = tag_values(tb, "train/actor/dvac_weight_mean")
    sq_mean = tag_values(tb, "train/actor/dvac_weight_sq_mean")
    weight_std = [math.sqrt(max(0.0, sq - avg * avg)) for avg, sq in zip(mean, sq_mean)]
    host = [float(row["host_available_gib"]) for row in resources]
    gpu_max = [float(row["gpu_max_gib"]) for row in resources]
    last180 = host[-180:] if len(host) >= 180 else host
    memory_three_hour_change = statistics.mean(last180[-30:]) - statistics.mean(last180[:30]) if len(last180) >= 60 else host[-1] - host[0]

    figures = [
        render_success(current, baseline, evals),
        render_optimization(current, tb),
        render_resources(resources),
    ]
    B.write_rows(SNAPSHOT / "dvac_v4_console_scalars_step1_31.csv", current)
    summary = {
        "capture": {"complete_step": 31, "incomplete_step": 32, "resource_rows": len(resources)},
        "success": {
            "dvac_latest": current_success[-1],
            "dvac_mean_step1_31": statistics.mean(current_success),
            "dvac_trailing5": statistics.mean(current_success[-5:]),
            "baseline_mean_step1_31": statistics.mean(baseline_success),
            "baseline_trailing5": statistics.mean(baseline_success[-5:]),
            "dvac_minus_baseline_mean": statistics.mean(current_success) - statistics.mean(baseline_success),
            "fixed64": evals,
        },
        "optimization_latest": {
            "approx_kl": current[-1]["approx_kl"],
            "clip_fraction": current[-1]["clip_fraction"],
            "grad_norm": current[-1]["grad_norm"],
            "policy_loss_abs": current[-1]["policy_loss_abs"],
        },
        "dvac_latest": {
            "weight_mean": mean[-1],
            "weight_std": weight_std[-1],
            "ess_fraction": tag_values(tb, "train/actor/dvac_weight_ess_fraction")[-1],
            "z_low_clip_fraction": tag_values(tb, "train/actor/dvac_z_low_clip_fraction")[-1],
            "z_high_clip_fraction": tag_values(tb, "train/actor/dvac_z_high_clip_fraction")[-1],
        },
        "timing": {
            "elapsed_s_step31": current[-1]["elapsed_s"],
            "median_step_s": statistics.median(float(row["step_time_s"]) for row in current),
            "latest5_mean_step_s": statistics.mean(float(row["step_time_s"]) for row in current[-5:]),
        },
        "resources": {
            "host_available_latest_gib": host[-1],
            "host_available_min_gib": min(host),
            "host_available_launch_gib": host[0],
            "three_hour_smoothed_change_gib": memory_three_hour_change,
            "gpu_card_peak_gib": max(gpu_max),
            "gpu_card_latest_max_gib": gpu_max[-1],
        },
        "figures": [str(path) for path in figures],
    }
    (SNAPSHOT / "summary_step31.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
