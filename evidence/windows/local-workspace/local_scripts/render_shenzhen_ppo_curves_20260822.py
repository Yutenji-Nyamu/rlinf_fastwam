from __future__ import annotations

import csv
import json
import math
import re
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"C:\Users\86136\Documents\rl")
EVIDENCE = ROOT / "docs" / "rlinf-shenzhen-pi0-ppo-rlt" / "evidence"
SOURCE = EVIDENCE / "ppo_formal100_metrics_live_step1_46_20260822.log"
CSV_OUT = EVIDENCE / "ppo_formal100_console_scalars_step1_46_20260822.csv"
SUCCESS_OUT = EVIDENCE / "ppo_formal100_success_eval_step1_46_20260822.png"
OPT_OUT = EVIDENCE / "ppo_formal100_optimization_step1_46_20260822.png"
TIME_OUT = EVIDENCE / "ppo_formal100_timing_step1_46_20260822.png"
RESOURCE_CSV = EVIDENCE / "ppo_formal100_resource_discrete_samples_through_step46_20260822.csv"
RESOURCE_OUT = EVIDENCE / "ppo_formal100_resource_discrete_through_step46_20260822.png"
EVAL_STEPS = {10, 20, 30, 40}

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

BLUE = "#2563EB"
GREEN = "#16803A"
ORANGE = "#EA580C"
RED = "#DC2626"
PURPLE = "#7C3AED"
GRAY = "#64748B"
GRID = "#D8E0EA"
INK = "#142033"
LIGHT = "#F4F7FB"
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
F_SUB = font(25)
F_AXIS = font(22)
F_SMALL = font(19)
F_PANEL = font(30, True)
F_LEGEND = font(21, True)


def first_number(segment: str, key: str) -> float:
    match = re.search(rf"(?<![\w/]){re.escape(key)}=({NUMBER})", segment)
    if not match:
        raise ValueError(f"missing {key}")
    return float(match.group(1))


def parse_metrics(path: Path) -> list[dict[str, float | int | None]]:
    text = ANSI_RE.sub("", path.read_text(encoding="utf-8", errors="replace"))
    matches = list(re.finditer(r"Global Step:\s*(\d+)/100", text))
    rows: list[dict[str, float | int | None]] = []
    for idx, match in enumerate(matches):
        step = int(match.group(1))
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        segment = text[match.start() : end]
        successes = [float(v) for v in re.findall(rf"(?<![\w/])success_once=({NUMBER})", segment)]
        if not successes:
            raise ValueError(f"step {step}: missing train success")
        row: dict[str, float | int | None] = {
            "step": step,
            "train_success": successes[0],
            "eval_success": successes[1] if len(successes) > 1 else None,
            "approx_kl": first_number(segment, "actor/approx_kl"),
            "clip_fraction": first_number(segment, "actor/clip_fraction"),
            "grad_norm": first_number(segment, "actor/grad_norm"),
            "critic_explained_variance": first_number(segment, "critic/explained_variance"),
            "step_time_s": first_number(segment, "step"),
            "generate_rollouts_s": first_number(segment, "generate_rollouts"),
            "actor_training_s": first_number(segment, "actor_training"),
        }
        rows.append(row)
    steps = [int(row["step"]) for row in rows]
    expected = list(range(1, max(steps) + 1))
    if steps != expected:
        raise ValueError(f"non-consecutive console steps: {steps}")
    for i, row in enumerate(rows):
        values = [float(row[key]) for key in row if key not in {"step", "eval_success"}]
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"non-finite scalar at step {i + 1}")
    return rows


def write_csv(rows: list[dict[str, float | int | None]]) -> None:
    keys = list(rows[0].keys())
    with CSV_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt_tick(value: float, span: float) -> str:
    if span <= 0.2:
        return f"{value:.3f}"
    if span <= 2:
        return f"{value:.2f}"
    if span <= 20:
        return f"{value:.1f}"
    return f"{value:.0f}"


def line_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    steps: list[int],
    series: list[tuple[str, list[float | None], str, int]],
    y_min: float,
    y_max: float,
    panel_title: str,
    eval_steps: set[int] | None = None,
    show_x_label: bool = True,
    zero_line: bool = False,
) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=20, fill="white", outline=GRID, width=2)
    plot_left, plot_top = left + 100, top + 78
    plot_right, plot_bottom = right - 38, bottom - (75 if show_x_label else 35)
    draw.text((left + 24, top + 20), panel_title, fill=INK, font=F_PANEL)
    x_min, x_max = min(steps), max(steps)

    def px(step: float) -> float:
        return plot_left + (step - x_min) / max(1, x_max - x_min) * (plot_right - plot_left)

    def py(value: float) -> float:
        return plot_bottom - (value - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

    if eval_steps:
        for step in sorted(eval_steps):
            x = px(step)
            width = max(7, (plot_right - plot_left) / max(1, x_max - x_min) * 0.65)
            draw.rectangle((x - width, plot_top, x + width, plot_bottom), fill=EVAL_SHADE)

    for i in range(6):
        value = y_min + (y_max - y_min) * i / 5
        y = py(value)
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=2)
        label = fmt_tick(value, y_max - y_min)
        bbox = draw.textbbox((0, 0), label, font=F_AXIS)
        draw.text((plot_left - 15 - (bbox[2] - bbox[0]), y - 13), label, fill=GRAY, font=F_AXIS)

    if zero_line and y_min < 0 < y_max:
        draw.line((plot_left, py(0), plot_right, py(0)), fill=GRAY, width=3)

    x_ticks = sorted({1, max(steps), *range(5, max(steps) + 1, 5)})
    for step in x_ticks:
        if step < x_min or step > x_max:
            continue
        x = px(step)
        draw.line((x, plot_bottom, x, plot_bottom + 9), fill=GRAY, width=2)
        label = str(step)
        bbox = draw.textbbox((0, 0), label, font=F_AXIS)
        draw.text((x - (bbox[2] - bbox[0]) / 2, plot_bottom + 13), label, fill=GRAY, font=F_AXIS)

    if show_x_label:
        label = "Completed global step"
        bbox = draw.textbbox((0, 0), label, font=F_AXIS)
        draw.text(((plot_left + plot_right - (bbox[2] - bbox[0])) / 2, bottom - 35), label, fill=GRAY, font=F_AXIS)

    legend_x = plot_left
    legend_y = top + 28
    title_box = draw.textbbox((0, 0), panel_title, font=F_PANEL)
    legend_x = max(legend_x, left + 38 + (title_box[2] - title_box[0]) + 45)
    for name, _, color, width in series:
        draw.line((legend_x, legend_y + 12, legend_x + 35, legend_y + 12), fill=color, width=width)
        draw.text((legend_x + 45, legend_y), name, fill=color, font=F_LEGEND)
        name_box = draw.textbbox((0, 0), name, font=F_LEGEND)
        legend_x += 45 + (name_box[2] - name_box[0]) + 28

    for _, values, color, width in series:
        points: list[tuple[float, float]] = []
        all_points: list[tuple[float, float]] = []
        for step, value in zip(steps, values):
            if value is None:
                if len(points) >= 2:
                    draw.line(points, fill=color, width=width, joint="curve")
                points = []
                continue
            point = (px(step), py(float(value)))
            points.append(point)
            all_points.append(point)
        if len(points) >= 2:
            draw.line(points, fill=color, width=width, joint="curve")
        marker_radius = 8 if any(value is None for value in values) else 4
        for x, y in all_points:
            draw.ellipse(
                (x - marker_radius, y - marker_radius, x + marker_radius, y + marker_radius),
                fill=color,
                outline="white" if marker_radius > 4 else color,
                width=3 if marker_radius > 4 else 1,
            )

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=GRAY, width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=GRAY, width=2)


def add_header(img: Image.Image, title: str, subtitle: str) -> ImageDraw.ImageDraw:
    draw = ImageDraw.Draw(img)
    draw.text((55, 35), title, fill=INK, font=F_TITLE)
    draw.text((57, 92), subtitle, fill=GRAY, font=F_SUB)
    return draw


def render_success(rows: list[dict[str, float | int | None]]) -> None:
    steps = [int(row["step"]) for row in rows]
    success = [float(row["train_success"]) for row in rows]
    mean5: list[float | None] = []
    for i in range(len(success)):
        mean5.append(statistics.mean(success[i - 4 : i + 1]) if i >= 4 else None)
    eval_values = [float(row["eval_success"]) if row["eval_success"] is not None else None for row in rows]
    img = Image.new("RGB", (1440, 1030), LIGHT)
    draw = add_header(
        img,
        "PPO train success and fixed-64 evaluation",
        "SZ-H100 | physical GPUs 4-7 | direct completed-step scalars | captured 2026-08-22 20:58 CST",
    )
    line_chart(
        draw,
        (45, 145, 1395, 965),
        steps,
        [
            ("Train success", success, BLUE, 4),
            ("5-step trailing mean", mean5, GREEN, 6),
            ("Fixed-64 eval", eval_values, ORANGE, 6),
        ],
        0,
        1,
        "Success rate",
        EVAL_STEPS,
    )
    img.save(SUCCESS_OUT, optimize=True)


def render_optimization(rows: list[dict[str, float | int | None]]) -> None:
    steps = [int(row["step"]) for row in rows]
    panels = [
        ("Approximate KL", "approx_kl", BLUE, 0.0, 0.08, False),
        ("PPO clip fraction", "clip_fraction", ORANGE, 0.0, 0.12, False),
        ("Pre-clip gradient norm", "grad_norm", PURPLE, 0.0, 50.0, False),
        ("Critic explained variance", "critic_explained_variance", GREEN, -0.3, 0.6, True),
    ]
    img = Image.new("RGB", (1440, 1900), LIGHT)
    draw = add_header(
        img,
        "PPO optimization and critic diagnostics",
        "All values are direct console scalars; no smoothing or interpolation",
    )
    top = 145
    height = 410
    gap = 25
    for idx, (title, key, color, low, high, zero) in enumerate(panels):
        values = [float(row[key]) for row in rows]
        line_chart(
            draw,
            (45, top + idx * (height + gap), 1395, top + idx * (height + gap) + height),
            steps,
            [(title, values, color, 5)],
            low,
            high,
            title,
            EVAL_STEPS,
            show_x_label=idx == len(panels) - 1,
            zero_line=zero,
        )
    img.save(OPT_OUT, optimize=True)


def render_timing(rows: list[dict[str, float | int | None]]) -> None:
    steps = [int(row["step"]) for row in rows]
    step_time = [float(row["step_time_s"]) for row in rows]
    rollout = [float(row["generate_rollouts_s"]) for row in rows]
    actor = [float(row["actor_training_s"]) for row in rows]
    img = Image.new("RGB", (1440, 1320), LIGHT)
    draw = add_header(
        img,
        "PPO wall-clock decomposition",
        "Eval/save cadence is every 10 steps; orange shading marks eval steps",
    )
    line_chart(
        draw,
        (45, 145, 1395, 760),
        steps,
        [("Whole step", step_time, RED, 5), ("Generate rollouts", rollout, BLUE, 5)],
        0,
        max(2100, math.ceil(max(step_time) / 100) * 100),
        "Step and rollout time (seconds)",
        EVAL_STEPS,
        show_x_label=False,
    )
    line_chart(
        draw,
        (45, 790, 1395, 1255),
        steps,
        [("Actor training", actor, GREEN, 5)],
        0,
        35,
        "Actor training time (seconds)",
        EVAL_STEPS,
    )
    img.save(TIME_OUT, optimize=True)


def render_resource() -> None:
    with RESOURCE_CSV.open(newline="", encoding="utf-8") as handle:
        samples = list(csv.DictReader(handle))
    steps = [int(row["completed_step"]) for row in samples]
    boundary_cgroup = [
        int(row["cgroup_memory_current_bytes"]) / (1024**3)
        if row["same_phase_boundary"] == "true"
        else None
        for row in samples
    ]
    latest_cgroup = [
        int(row["cgroup_memory_current_bytes"]) / (1024**3)
        if row["same_phase_boundary"] == "false"
        else None
        for row in samples
    ]
    boundary_available = [
        int(row["host_mem_available_kib"]) / (1024**2)
        if row["same_phase_boundary"] == "true"
        else None
        for row in samples
    ]
    latest_available = [
        int(row["host_mem_available_kib"]) / (1024**2)
        if row["same_phase_boundary"] == "false"
        else None
        for row in samples
    ]

    img = Image.new("RGB", (1440, 1580), LIGHT)
    draw = add_header(
        img,
        "PPO resource pressure: discrete read-only samples",
        "Boundary points are comparable; the latest red point is Step 47 rollout 2/4, not a boundary",
    )
    line_chart(
        draw,
        (45, 145, 1395, 660),
        steps,
        [
            ("Comparable boundaries", boundary_cgroup, PURPLE, 6),
            ("Latest mid-rollout", latest_cgroup, RED, 6),
        ],
        1300,
        1900,
        "Training cgroup memory (GiB)",
        show_x_label=False,
    )
    line_chart(
        draw,
        (45, 690, 1395, 1205),
        steps,
        [
            ("Comparable boundaries", boundary_available, GREEN, 6),
            ("Latest mid-rollout", latest_available, RED, 6),
        ],
        0,
        650,
        "Host MemAvailable (GiB)",
    )

    left, top, right, bottom = 45, 1235, 1395, 1525
    draw.rounded_rectangle((left, top, right, bottom), radius=20, fill="white", outline=GRID, width=2)
    draw.text((left + 24, top + 20), "Latest GPU memory snapshot", fill=INK, font=F_PANEL)
    draw.text((left + 635, top + 27), "20:58 CST | physical GPUs 4-7 | 81,559 MiB each", fill=GRAY, font=F_SMALL)
    latest = samples[-1]
    values = [int(latest[f"gpu{gpu}_mib"]) for gpu in range(4, 8)]
    chart_left, chart_right = left + 120, right - 45
    chart_top, chart_bottom = top + 92, bottom - 48
    bar_gap = 55
    bar_width = (chart_right - chart_left - 3 * bar_gap) / 4
    for idx, (gpu, value) in enumerate(zip(range(4, 8), values)):
        x0 = chart_left + idx * (bar_width + bar_gap)
        x1 = x0 + bar_width
        y0 = chart_bottom - value / 81559 * (chart_bottom - chart_top)
        draw.rounded_rectangle((x0, chart_top, x1, chart_bottom), radius=12, fill="#EDF2F8")
        draw.rounded_rectangle((x0, y0, x1, chart_bottom), radius=12, fill=BLUE)
        label = f"GPU {gpu}"
        value_label = f"{value / 1024:.1f} GiB"
        draw.text((x0 + 10, chart_bottom + 9), label, fill=INK, font=F_SMALL)
        draw.text((x0 + 10, max(chart_top + 4, y0 - 28)), value_label, fill=INK, font=F_SMALL)
    img.save(RESOURCE_OUT, optimize=True)


def main() -> None:
    rows = parse_metrics(SOURCE)
    if len(rows) != 46:
        raise ValueError(f"expected 46 complete steps, got {len(rows)}")
    write_csv(rows)
    render_success(rows)
    render_optimization(rows)
    render_timing(rows)
    render_resource()

    success = [float(row["train_success"]) for row in rows]
    summary = {
        "steps": [1, len(rows)],
        "train_success": {
            "first": success[0],
            "latest": success[-1],
            "mean": statistics.mean(success),
            "trailing5_latest": statistics.mean(success[-5:]),
            "min": min(success),
            "max": max(success),
        },
        "eval": {int(row["step"]): float(row["eval_success"]) for row in rows if row["eval_success"] is not None},
        "latest": {key: rows[-1][key] for key in rows[-1] if key != "eval_success"},
        "timing_median_s": {
            "step": statistics.median(float(row["step_time_s"]) for row in rows),
            "rollout": statistics.median(float(row["generate_rollouts_s"]) for row in rows),
            "actor": statistics.median(float(row["actor_training_s"]) for row in rows),
        },
        "outputs": [str(CSV_OUT), str(SUCCESS_OUT), str(OPT_OUT), str(TIME_OUT), str(RESOURCE_OUT)],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
