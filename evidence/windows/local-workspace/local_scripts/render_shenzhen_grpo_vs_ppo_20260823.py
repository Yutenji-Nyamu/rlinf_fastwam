from __future__ import annotations

import csv
import json
import math
import re
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"C:\Users\86136\Documents\rl")
EVIDENCE = ROOT / "docs" / "rlinf-shenzhen-pi0-ppo-rlt" / "evidence"
SNAPSHOT = EVIDENCE / "grpo_v2_live_step27_20260823"
GRPO_LOG = SNAPSHOT / "metrics.log"
RESOURCE_CSV = SNAPSHOT / "resource.csv"
PPO_CSV = EVIDENCE / "ppo_formal100_console_scalars_step1_46_20260822.csv"
PPO_RESOURCE_CSV = EVIDENCE / "ppo_formal100_resource_discrete_samples_through_step46_20260822.csv"

SCALARS_OUT = SNAPSHOT / "grpo_console_scalars_step1_27.csv"
RESOURCE_STEP_OUT = SNAPSHOT / "grpo_resource_nearest_minute_by_completed_step.csv"
SUCCESS_OUT = SNAPSHOT / "grpo_vs_ppo_success_step1_27.png"
OPT_OUT = SNAPSHOT / "grpo_vs_ppo_optimization_timing_step1_27.png"
RESOURCE_OUT = SNAPSHOT / "grpo_resource_and_ppo_comparison_step27_capture.png"
SUMMARY_OUT = SNAPSHOT / "summary_step27.json"

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

INK = "#142033"
GRAY = "#64748B"
GRID = "#D8E0EA"
LIGHT = "#F4F7FB"
WHITE = "#FFFFFF"
GRPO = "#2563EB"
GRPO_LIGHT = "#93C5FD"
PPO = "#EA580C"
PPO_LIGHT = "#FDBA74"
GREEN = "#16803A"
PURPLE = "#7C3AED"
RED = "#DC2626"
EVAL_SHADE = "#FFF5E8"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


F_TITLE = font(42, True)
F_SUB = font(24)
F_PANEL = font(29, True)
F_AXIS = font(21)
F_SMALL = font(18)
F_LEGEND = font(20, True)
F_CARD = font(25, True)


def first_number(segment: str, key: str) -> float:
    match = re.search(rf"(?<![\w/]){re.escape(key)}=({NUMBER})", segment)
    if not match:
        raise ValueError(f"missing {key}")
    return float(match.group(1))


def parse_elapsed(value: str) -> float:
    parts = [int(item) for item in value.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"unexpected elapsed value: {value}")


def parse_grpo(path: Path) -> list[dict[str, float | int | None]]:
    text = ANSI_RE.sub("", path.read_text(encoding="utf-8", errors="replace"))
    matches = list(re.finditer(r"Global Step:\s*(\d+)/100", text))
    rows: list[dict[str, float | int | None]] = []
    for idx, match in enumerate(matches):
        step = int(match.group(1))
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        segment = text[match.start() : end]
        success = [float(value) for value in re.findall(rf"(?<![\w/])success_once=({NUMBER})", segment)]
        elapsed_match = re.search(r"Elapsed:\s*([0-9:]+)", segment)
        if not success or elapsed_match is None:
            raise ValueError(f"step {step}: incomplete table")
        row: dict[str, float | int | None] = {
            "step": step,
            "elapsed_s": parse_elapsed(elapsed_match.group(1)),
            "train_success": success[0],
            "eval_success": success[1] if len(success) > 1 else None,
            "approx_kl": first_number(segment, "actor/approx_kl"),
            "clip_fraction": first_number(segment, "actor/clip_fraction"),
            "grad_norm": first_number(segment, "actor/grad_norm"),
            "ratio_abs": first_number(segment, "actor/ratio_abs"),
            "policy_loss_abs": first_number(segment, "actor/policy_loss_abs"),
            "step_time_s": first_number(segment, "step"),
            "generate_rollouts_s": first_number(segment, "generate_rollouts"),
            "actor_training_s": first_number(segment, "actor_training"),
        }
        rows.append(row)
    steps = [int(row["step"]) for row in rows]
    if not steps or steps != list(range(1, max(steps) + 1)):
        raise ValueError(f"non-consecutive GRPO steps: {steps}")
    for row in rows:
        for key, value in row.items():
            if key == "eval_success" and value is None:
                continue
            if not math.isfinite(float(value)):
                raise ValueError(f"non-finite {key} at step {row['step']}")
    return rows


def read_ppo(path: Path, through: int) -> list[dict[str, float | int | None]]:
    rows: list[dict[str, float | int | None]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            step = int(raw["step"])
            if step > through:
                continue
            rows.append(
                {
                    "step": step,
                    "train_success": float(raw["train_success"]),
                    "eval_success": float(raw["eval_success"]) if raw["eval_success"] else None,
                    "approx_kl": float(raw["approx_kl"]),
                    "clip_fraction": float(raw["clip_fraction"]),
                    "grad_norm": float(raw["grad_norm"]),
                    "step_time_s": float(raw["step_time_s"]),
                    "generate_rollouts_s": float(raw["generate_rollouts_s"]),
                    "actor_training_s": float(raw["actor_training_s"]),
                }
            )
    if [int(row["step"]) for row in rows] != list(range(1, through + 1)):
        raise ValueError("PPO comparison rows are not consecutive")
    return rows


def trailing(values: list[float], window: int = 5) -> list[float | None]:
    return [statistics.mean(values[idx - window + 1 : idx + 1]) if idx >= window - 1 else None for idx in range(len(values))]


def write_rows(path: Path, rows: list[dict[str, float | int | None]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_header(img: Image.Image, title: str, subtitle: str) -> ImageDraw.ImageDraw:
    draw = ImageDraw.Draw(img)
    draw.text((55, 34), title, fill=INK, font=F_TITLE)
    draw.text((57, 91), subtitle, fill=GRAY, font=F_SUB)
    return draw


def fmt_tick(value: float, span: float) -> str:
    if span <= 0.2:
        return f"{value:.2f}"
    if span <= 2:
        return f"{value:.1f}"
    if span <= 30:
        return f"{value:.1f}"
    return f"{value:.0f}"


def line_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    xs: list[float],
    series: list[tuple[str, list[float | None], str, int]],
    y_min: float,
    y_max: float,
    title: str,
    *,
    x_label: str = "Completed global step",
    x_ticks: list[float] | None = None,
    x_tick_format=lambda value: f"{value:g}",
    eval_steps: set[int] | None = None,
    zero_line: bool = False,
) -> tuple[float, float, float, float]:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=20, fill=WHITE, outline=GRID, width=2)
    plot_left, plot_top = left + 105, top + 78
    plot_right, plot_bottom = right - 38, bottom - 72
    draw.text((left + 24, top + 20), title, fill=INK, font=F_PANEL)
    x_min, x_max = min(xs), max(xs)

    def px(value: float) -> float:
        return plot_left + (value - x_min) / max(1e-9, x_max - x_min) * (plot_right - plot_left)

    def py(value: float) -> float:
        return plot_bottom - (value - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

    if eval_steps:
        for step in sorted(eval_steps):
            if x_min <= step <= x_max:
                x = px(step)
                width = max(7, (plot_right - plot_left) / max(1, x_max - x_min) * 0.6)
                draw.rectangle((x - width, plot_top, x + width, plot_bottom), fill=EVAL_SHADE)

    for idx in range(6):
        value = y_min + (y_max - y_min) * idx / 5
        y = py(value)
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=2)
        label = fmt_tick(value, y_max - y_min)
        bbox = draw.textbbox((0, 0), label, font=F_AXIS)
        draw.text((plot_left - 14 - (bbox[2] - bbox[0]), y - 12), label, fill=GRAY, font=F_AXIS)

    if zero_line and y_min < 0 < y_max:
        draw.line((plot_left, py(0), plot_right, py(0)), fill=GRAY, width=3)

    if x_ticks is None:
        integer_max = int(math.floor(x_max))
        x_ticks = sorted({x_min, x_max, *range(5, integer_max + 1, 5)})
    for value in x_ticks:
        if not (x_min <= value <= x_max):
            continue
        x = px(value)
        draw.line((x, plot_bottom, x, plot_bottom + 8), fill=GRAY, width=2)
        label = x_tick_format(value)
        bbox = draw.textbbox((0, 0), label, font=F_AXIS)
        draw.text((x - (bbox[2] - bbox[0]) / 2, plot_bottom + 12), label, fill=GRAY, font=F_AXIS)

    label_box = draw.textbbox((0, 0), x_label, font=F_AXIS)
    draw.text(((plot_left + plot_right - (label_box[2] - label_box[0])) / 2, bottom - 33), x_label, fill=GRAY, font=F_AXIS)

    legend_x = plot_left
    legend_y = top + 28
    title_box = draw.textbbox((0, 0), title, font=F_PANEL)
    legend_x = max(legend_x, left + 38 + (title_box[2] - title_box[0]) + 35)
    for name, _, color, width in series:
        draw.line((legend_x, legend_y + 12, legend_x + 32, legend_y + 12), fill=color, width=width)
        draw.text((legend_x + 41, legend_y), name, fill=color, font=F_LEGEND)
        name_box = draw.textbbox((0, 0), name, font=F_LEGEND)
        legend_x += 41 + (name_box[2] - name_box[0]) + 24

    for _, values, color, width in series:
        points: list[tuple[float, float]] = []
        for x_value, y_value in zip(xs, values):
            if y_value is None:
                if len(points) >= 2:
                    draw.line(points, fill=color, width=width, joint="curve")
                points = []
                continue
            point = (px(x_value), py(float(y_value)))
            points.append(point)
            radius = 4 if width <= 3 else 5
            draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill=color)
        if len(points) >= 2:
            draw.line(points, fill=color, width=width, joint="curve")

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=GRAY, width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=GRAY, width=2)
    return plot_left, plot_top, plot_right, plot_bottom


def draw_eval_markers(
    draw: ImageDraw.ImageDraw,
    plot: tuple[float, float, float, float],
    xs: list[float],
    y_min: float,
    y_max: float,
    evals: list[tuple[int, float | None]],
    color: str,
    shape: str,
) -> None:
    left, top, right, bottom = plot
    x_min, x_max = min(xs), max(xs)
    for step, value in evals:
        if value is None or step < x_min or step > x_max:
            continue
        x = left + (step - x_min) / max(1e-9, x_max - x_min) * (right - left)
        y = bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
        if shape == "square":
            draw.rectangle((x - 10, y - 10, x + 10, y + 10), fill=color, outline=WHITE, width=3)
        else:
            draw.polygon(((x, y - 12), (x - 12, y + 10), (x + 12, y + 10)), fill=color)


def render_success(grpo: list[dict], ppo: list[dict]) -> None:
    steps = [float(row["step"]) for row in grpo]
    grpo_success = [float(row["train_success"]) for row in grpo]
    ppo_success = [float(row["train_success"]) for row in ppo]
    grpo_ma = trailing(grpo_success)
    ppo_ma = trailing(ppo_success)
    img = Image.new("RGB", (1440, 1360), LIGHT)
    draw = add_header(
        img,
        "GRPO vs PPO: training success",
        "Matched SZ-H100 budget: 4 GPUs, 512 trajectories/step, fixed-64 eval every 10 steps",
    )
    line_panel(
        draw,
        (45, 145, 1395, 705),
        steps,
        [("GRPO raw", grpo_success, GRPO_LIGHT, 3), ("PPO raw", ppo_success, PPO_LIGHT, 3)],
        0.65,
        1.0,
        "Per-step rollout success",
        eval_steps={10, 20},
    )
    plot = line_panel(
        draw,
        (45, 735, 1395, 1295),
        steps,
        [("GRPO trailing-5", grpo_ma, GRPO, 7), ("PPO trailing-5", ppo_ma, PPO, 7)],
        0.65,
        1.0,
        "Five-step mean and fixed-64 evaluation",
        eval_steps={10, 20},
    )
    draw_eval_markers(draw, plot, steps, 0.65, 1.0, [(int(row["step"]), row["eval_success"]) for row in grpo], GRPO, "square")
    draw_eval_markers(draw, plot, steps, 0.65, 1.0, [(int(row["step"]), row["eval_success"]) for row in ppo], PPO, "triangle")
    draw.text((102, 1258), "Squares: GRPO fixed-64   Triangles: PPO fixed-64", fill=GRAY, font=F_SMALL)
    img.save(SUCCESS_OUT, optimize=True)


def render_optimization(grpo: list[dict], ppo: list[dict]) -> None:
    steps = [float(row["step"]) for row in grpo]
    panels = [
        ("Approximate KL", "approx_kl", 0.0, 0.08),
        ("Clip fraction", "clip_fraction", 0.0, 0.14),
        ("Pre-clip gradient norm", "grad_norm", 0.0, 45.0),
        ("Whole-step wall time", "step_time_s", 0.0, 2100.0),
    ]
    img = Image.new("RGB", (1440, 1950), LIGHT)
    draw = add_header(
        img,
        "GRPO vs PPO: optimizer shape and time",
        "Direct complete-step console scalars; actor-only GRPO has no critic explained-variance series",
    )
    top, height, gap = 145, 410, 28
    for idx, (title, key, low, high) in enumerate(panels):
        line_panel(
            draw,
            (45, top + idx * (height + gap), 1395, top + idx * (height + gap) + height),
            steps,
            [
                ("GRPO", [float(row[key]) for row in grpo], GRPO, 5),
                ("PPO", [float(row[key]) for row in ppo], PPO, 5),
            ],
            low,
            high,
            title,
            eval_steps={10, 20},
        )
    img.save(OPT_OUT, optimize=True)


def read_resources(path: Path) -> list[dict[str, float | datetime]]:
    rows: list[dict[str, float | datetime]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            gpus = [float(raw[f"gpu{gpu}_used_mib"]) / 1024 for gpu in range(4, 8)]
            rows.append(
                {
                    "timestamp": datetime.fromisoformat(raw["timestamp"]),
                    "cgroup_gib": float(raw["cgroup_memory_current_bytes"]) / 1024**3,
                    "host_available_gib": float(raw["host_mem_available_kib"]) / 1024**2,
                    "gpu_mean_gib": statistics.mean(gpus),
                    "gpu_max_gib": max(gpus),
                }
            )
    return rows


def align_resources(grpo: list[dict], resources: list[dict]) -> list[dict[str, float | int | str]]:
    start = resources[0]["timestamp"]
    assert isinstance(start, datetime)
    output: list[dict[str, float | int | str]] = []
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
    return output


def read_ppo_resource(path: Path) -> list[dict[str, float | int | bool]]:
    rows: list[dict[str, float | int | bool]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "step": int(raw["completed_step"]),
                    "boundary": raw["same_phase_boundary"].lower() == "true",
                    "cgroup_gib": float(raw["cgroup_memory_current_bytes"]) / 1024**3,
                    "host_available_gib": float(raw["host_mem_available_kib"]) / 1024**2,
                }
            )
    return rows


def render_resource(resources: list[dict], aligned: list[dict], ppo_resource: list[dict]) -> None:
    start = resources[0]["timestamp"]
    elapsed_h = [(row["timestamp"] - start).total_seconds() / 3600 for row in resources]
    last_h = elapsed_h[-1]
    x_ticks = [value for value in range(0, math.ceil(last_h) + 1, 2)]
    if last_h not in x_ticks:
        x_ticks.append(last_h)
    img = Image.new("RGB", (1440, 2040), LIGHT)
    capture_cst = resources[-1]["timestamp"].astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M CST")
    draw = add_header(
        img,
        "GRPO resource timeline and PPO reference",
        f"One-minute observer samples through {capture_cst}; live swap is reported separately",
    )
    line_panel(
        draw,
        (45, 145, 1395, 660),
        elapsed_h,
        [
            ("GRPO cgroup used", [float(row["cgroup_gib"]) for row in resources], PURPLE, 5),
            ("Host available", [float(row["host_available_gib"]) for row in resources], GREEN, 5),
        ],
        0,
        2100,
        "Main-memory pressure",
        x_label="Hours since observer start",
        x_ticks=x_ticks,
        x_tick_format=lambda value: f"{value:.1f}" if value == last_h else f"{value:.0f}",
    )
    line_panel(
        draw,
        (45, 690, 1395, 1205),
        elapsed_h,
        [
            ("Mean GPU memory", [float(row["gpu_mean_gib"]) for row in resources], GRPO, 5),
            ("Max card memory", [float(row["gpu_max_gib"]) for row in resources], RED, 5),
        ],
        0,
        80,
        "Physical GPUs 4-7 memory",
        x_label="Hours since observer start",
        x_ticks=x_ticks,
        x_tick_format=lambda value: f"{value:.1f}" if value == last_h else f"{value:.0f}",
    )

    grpo_steps = [float(row["step"]) for row in aligned]
    grpo_cgroup = [float(row["cgroup_gib"]) for row in aligned]
    ppo_boundary = [row for row in ppo_resource if bool(row["boundary"])]
    ppo_x = [float(row["step"]) for row in ppo_boundary]
    ppo_y = [float(row["cgroup_gib"]) for row in ppo_boundary]
    combined_x = sorted({*grpo_steps, *ppo_x})
    ppo_on_combined: list[float | None] = [ppo_y[ppo_x.index(x)] if x in ppo_x else None for x in combined_x]
    grpo_on_combined: list[float | None] = [grpo_cgroup[grpo_steps.index(x)] if x in grpo_steps else None for x in combined_x]
    plot = line_panel(
        draw,
        (45, 1235, 1395, 1750),
        combined_x,
        [("GRPO nearest-minute", grpo_on_combined, GRPO, 5), ("PPO read-only boundaries", ppo_on_combined, PPO, 6)],
        0,
        2000,
        "Cgroup memory by completed step",
        x_ticks=[1, 5, 10, 15, 20, 25, 30, 35, 40, 46],
    )
    # PPO points are intentionally disconnected by the generic None-gap behavior.
    left, top, right, bottom = 45, 1780, 1395, 1985
    draw.rounded_rectangle((left, top, right, bottom), radius=20, fill=WHITE, outline=GRID, width=2)
    latest = resources[-1]
    draw.text((left + 24, top + 20), "Latest captured state", fill=INK, font=F_PANEL)
    cards = [
        ("Cgroup", f"{float(latest['cgroup_gib']):.1f} GiB"),
        ("Host available", f"{float(latest['host_available_gib']):.1f} GiB"),
        ("GPU mean", f"{float(latest['gpu_mean_gib']):.1f} GiB/card"),
        ("GPU max", f"{float(latest['gpu_max_gib']):.1f} GiB"),
    ]
    for idx, (label, value) in enumerate(cards):
        x = left + 35 + idx * 323
        draw.text((x, top + 88), label, fill=GRAY, font=F_SMALL)
        draw.text((x, top + 120), value, fill=INK, font=F_CARD)
    img.save(RESOURCE_OUT, optimize=True)


def window_mean(rows: list[dict], start: int, end: int) -> float:
    values = [float(row["train_success"]) for row in rows if start <= int(row["step"]) <= end]
    return statistics.mean(values)


def summarize(grpo: list[dict], ppo: list[dict], resources: list[dict], aligned: list[dict], ppo_resource: list[dict]) -> dict:
    grpo_success = [float(row["train_success"]) for row in grpo]
    ppo_success = [float(row["train_success"]) for row in ppo]
    latest_resource = resources[-1]
    aligned_map = {int(row["step"]): row for row in aligned}
    ppo_boundary_map = {int(row["step"]): row for row in ppo_resource if bool(row["boundary"])}
    comparisons: dict[str, dict] = {}
    for step in (22, 26):
        comparisons[str(step)] = {
            "grpo_cgroup_gib_nearest_minute": aligned_map[step]["cgroup_gib"],
            "ppo_cgroup_gib_boundary": round(float(ppo_boundary_map[step]["cgroup_gib"]), 3),
            "grpo_minus_ppo_gib": round(
                float(aligned_map[step]["cgroup_gib"]) - float(ppo_boundary_map[step]["cgroup_gib"]), 3
            ),
        }
    return {
        "capture": {
            "complete_steps": len(grpo),
            "resource_first_utc": resources[0]["timestamp"].isoformat(),
            "resource_last_utc": resources[-1]["timestamp"].isoformat(),
            "resource_rows": len(resources),
        },
        "success": {
            "grpo": {
                "first": grpo_success[0],
                "latest": grpo_success[-1],
                "mean_step1_27": statistics.mean(grpo_success),
                "trailing5_latest": statistics.mean(grpo_success[-5:]),
                "block_1_10": window_mean(grpo, 1, 10),
                "block_11_20": window_mean(grpo, 11, 20),
                "block_21_27": window_mean(grpo, 21, 27),
                "eval": {str(row["step"]): row["eval_success"] for row in grpo if row["eval_success"] is not None},
            },
            "ppo_matched_steps": {
                "first": ppo_success[0],
                "latest": ppo_success[-1],
                "mean_step1_27": statistics.mean(ppo_success),
                "trailing5_latest": statistics.mean(ppo_success[-5:]),
                "block_1_10": window_mean(ppo, 1, 10),
                "block_11_20": window_mean(ppo, 11, 20),
                "block_21_27": window_mean(ppo, 21, 27),
                "eval": {str(row["step"]): row["eval_success"] for row in ppo if row["eval_success"] is not None},
            },
            "grpo_minus_ppo": {
                "mean_step1_27": statistics.mean(grpo_success) - statistics.mean(ppo_success),
                "trailing5_latest": statistics.mean(grpo_success[-5:]) - statistics.mean(ppo_success[-5:]),
            },
        },
        "optimization": {
            "grpo_latest": {key: grpo[-1][key] for key in ("approx_kl", "clip_fraction", "grad_norm", "ratio_abs")},
            "grpo_ranges": {
                key: [min(float(row[key]) for row in grpo), max(float(row[key]) for row in grpo)]
                for key in ("approx_kl", "clip_fraction", "grad_norm", "ratio_abs")
            },
            "ppo_latest_matched_step": {key: ppo[-1][key] for key in ("approx_kl", "clip_fraction", "grad_norm")},
        },
        "timing_median_s": {
            "grpo_step": statistics.median(float(row["step_time_s"]) for row in grpo),
            "grpo_rollout": statistics.median(float(row["generate_rollouts_s"]) for row in grpo),
            "grpo_actor": statistics.median(float(row["actor_training_s"]) for row in grpo),
            "ppo_step_matched": statistics.median(float(row["step_time_s"]) for row in ppo),
            "ppo_rollout_matched": statistics.median(float(row["generate_rollouts_s"]) for row in ppo),
            "ppo_actor_matched": statistics.median(float(row["actor_training_s"]) for row in ppo),
        },
        "resources": {
            "latest": {
                "cgroup_gib": latest_resource["cgroup_gib"],
                "host_available_gib": latest_resource["host_available_gib"],
                "gpu_mean_gib": latest_resource["gpu_mean_gib"],
                "gpu_max_gib": latest_resource["gpu_max_gib"],
            },
            "peak": {
                "cgroup_gib": max(float(row["cgroup_gib"]) for row in resources),
                "gpu_max_gib": max(float(row["gpu_max_gib"]) for row in resources),
            },
            "grpo_vs_ppo_at_common_boundaries": comparisons,
        },
        "outputs": [str(path) for path in (SCALARS_OUT, RESOURCE_STEP_OUT, SUCCESS_OUT, OPT_OUT, RESOURCE_OUT)],
    }


def main() -> None:
    grpo = parse_grpo(GRPO_LOG)
    if len(grpo) != 27:
        raise ValueError(f"snapshot expected 27 complete steps, got {len(grpo)}")
    ppo = read_ppo(PPO_CSV, len(grpo))
    resources = read_resources(RESOURCE_CSV)
    aligned = align_resources(grpo, resources)
    ppo_resource = read_ppo_resource(PPO_RESOURCE_CSV)
    write_rows(SCALARS_OUT, grpo)
    write_rows(RESOURCE_STEP_OUT, aligned)
    render_success(grpo, ppo)
    render_optimization(grpo, ppo)
    render_resource(resources, aligned, ppo_resource)
    summary = summarize(grpo, ppo, resources, aligned, ppo_resource)
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
