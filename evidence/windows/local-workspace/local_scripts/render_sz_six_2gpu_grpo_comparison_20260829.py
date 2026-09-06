from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import statistics
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(r"C:\Users\86136\Documents\rl")
SNAPSHOT = os.environ.get(
    "SZ_GRPO_SNAPSHOT", "six-2gpu-grpo-comparison-live-20260829"
)
OUT = (
    ROOT
    / "docs"
    / "rlinf-shenzhen-grpo-dvac-action-adv"
    / "evidence"
    / SNAPSHOT
)
BASE_SCRIPT = ROOT / "local_scripts" / "render_shenzhen_grpo_vs_ppo_20260823.py"
TITLE = "SZ-H100 matched two-GPU GRPO family — all formal trajectories"
SUBTITLE = "Common budget: 64 train env x 4 rollout epochs, G8, B1024; lines stop at each run's real final/latest step."

# Okabe-Ito-derived colors plus black; line patterns and markers remain distinct
# even when colors are viewed on a small screen.
RUNS = {
    "control": {
        "name": "GRPO Control",
        "color": "#111827",
        "pattern": None,
        "marker": "circle",
        "path": ROOT
        / "docs/rlinf-shenzhen-grpo-dvac-action-adv/evidence/grpo-control-2gpu-stopped-step96-20260828/raw/runtime/driver.log",
    },
    "st_02": {
        "name": "ST-DVAC [0,2]",
        "color": "#E69F00",
        "pattern": (18, 8),
        "marker": "square",
        "path": ROOT
        / "docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/shenzhen_grpo_dvac_w0to2_2gpu_stopped_step52_light_evidence_20260827/runtime/driver.log",
    },
    "prism": {
        "name": "Prism-RLOO",
        "color": "#009E73",
        "pattern": (7, 6),
        "marker": "triangle",
        "path": OUT.parent
        / "action-adv-prism-stopped-20260828/raw/prism/runtime/driver.log",
    },
    "action_old": {
        "name": "Action-Adv old (H-mean)",
        "color": "#6B7280",
        "pattern": (18, 6, 4, 6),
        "marker": "cross",
        "path": OUT.parent
        / "action-adv-prism-stopped-20260828/raw/action_adv/runtime/driver.log",
    },
    "action_fix": {
        "name": "Action-Adv Fix [0,2]",
        "color": "#CC79A7",
        "pattern": None,
        "marker": "diamond",
        "path": OUT / "raw/action_adv_fix/runtime/driver.log",
    },
    "st_half": {
        "name": "ST-DVAC [0.5,1.5]",
        "color": "#0072B2",
        "pattern": (12, 7),
        "marker": "star",
        "path": OUT / "raw/st_half/runtime/driver.log",
    },
}


def load_base():
    spec = importlib.util.spec_from_file_location("plot_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("plot helper unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def draw_patterned_line(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    *,
    fill: str,
    width: int,
    pattern: tuple[int, ...] | None,
) -> None:
    if len(points) < 2:
        return
    if pattern is None:
        draw.line(points, fill=fill, width=width, joint="curve")
        return
    phase = 0
    remaining = float(pattern[0])
    drawing = True
    for start, end in zip(points, points[1:]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length == 0:
            continue
        travelled = 0.0
        while travelled < length:
            take = min(remaining, length - travelled)
            a = travelled / length
            b = (travelled + take) / length
            p0 = (start[0] + dx * a, start[1] + dy * a)
            p1 = (start[0] + dx * b, start[1] + dy * b)
            if drawing:
                draw.line((p0, p1), fill=fill, width=width)
            travelled += take
            remaining -= take
            if remaining <= 1e-9:
                phase = (phase + 1) % len(pattern)
                remaining = float(pattern[phase])
                drawing = not drawing


def draw_marker(draw: ImageDraw.ImageDraw, x: float, y: float, color: str, shape: str, radius: int = 6) -> None:
    if shape == "circle":
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="white", width=2)
    elif shape == "square":
        draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=color, outline="white", width=2)
    elif shape == "triangle":
        draw.polygon(((x, y - radius - 2), (x - radius - 1, y + radius), (x + radius + 1, y + radius)), fill=color)
    elif shape == "diamond":
        draw.polygon(((x, y - radius - 1), (x - radius - 1, y), (x, y + radius + 1), (x + radius + 1, y)), fill=color)
    elif shape == "cross":
        draw.line((x - radius, y - radius, x + radius, y + radius), fill=color, width=3)
        draw.line((x - radius, y + radius, x + radius, y - radius), fill=color, width=3)
    else:
        draw.line((x - radius, y, x + radius, y), fill=color, width=3)
        draw.line((x, y - radius, x, y + radius), fill=color, width=3)
        draw.line((x - radius + 2, y - radius + 2, x + radius - 2, y + radius - 2), fill=color, width=2)
        draw.line((x - radius + 2, y + radius - 2, x + radius - 2, y - radius + 2), fill=color, width=2)


def draw_shared_legend(draw: ImageDraw.ImageDraw, base, data: dict[str, dict]) -> None:
    x0, y0 = 70, 118
    col_w, row_h = 555, 44
    for idx, (key, meta) in enumerate(RUNS.items()):
        row, col = divmod(idx, 3)
        x, y = x0 + col * col_w, y0 + row * row_h
        draw_patterned_line(
            draw,
            [(x, y + 13), (x + 52, y + 13)],
            fill=meta["color"],
            width=5,
            pattern=meta["pattern"],
        )
        draw_marker(draw, x + 26, y + 13, meta["color"], meta["marker"], 5)
        label = f"{meta['name']}  (Step {data[key]['steps'][-1]})"
        draw.text((x + 65, y), label, fill=meta["color"], font=base.F_LEGEND)


def panel(
    draw: ImageDraw.ImageDraw,
    base,
    box: tuple[int, int, int, int],
    data: dict[str, dict],
    metric: str,
    title: str,
    y_min: float,
    y_max: float,
    maximum: int,
) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=20, fill=base.WHITE, outline=base.GRID, width=2)
    plot_left, plot_top = left + 104, top + 67
    plot_right, plot_bottom = right - 34, bottom - 60
    draw.text((left + 24, top + 18), title, fill=base.INK, font=base.F_PANEL)

    def px(step: float) -> float:
        return plot_left + (step - 1) / max(1, maximum - 1) * (plot_right - plot_left)

    def py(value: float) -> float:
        return plot_bottom - (value - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

    for idx in range(6):
        value = y_min + (y_max - y_min) * idx / 5
        y = py(value)
        draw.line((plot_left, y, plot_right, y), fill=base.GRID, width=2)
        label = f"{value * 100:.0f}%"
        bbox = draw.textbbox((0, 0), label, font=base.F_AXIS)
        draw.text((plot_left - 14 - (bbox[2] - bbox[0]), y - 12), label, fill=base.GRAY, font=base.F_AXIS)

    ticks = [1, *range(10, maximum + 1, 10)]
    if maximum not in ticks:
        ticks.append(maximum)
    for step in ticks:
        x = px(step)
        draw.line((x, plot_bottom, x, plot_bottom + 8), fill=base.GRAY, width=2)
        label = str(step)
        bbox = draw.textbbox((0, 0), label, font=base.F_AXIS)
        draw.text((x - (bbox[2] - bbox[0]) / 2, plot_bottom + 11), label, fill=base.GRAY, font=base.F_AXIS)

    for key, meta in RUNS.items():
        points: list[tuple[float, float]] = []
        for step, value in zip(data[key]["steps"], data[key][metric]):
            if value is not None:
                points.append((px(step), py(float(value))))
        draw_patterned_line(draw, points, fill=meta["color"], width=4, pattern=meta["pattern"])
        marker_steps = {step for step in data[key]["steps"] if step % 10 == 0}
        marker_steps.add(data[key]["steps"][-1])
        if metric == "fixed":
            marker_steps.update(
                step
                for step, value in zip(data[key]["steps"], data[key][metric])
                if value is not None
            )
        for step, value in zip(data[key]["steps"], data[key][metric]):
            if value is not None and step in marker_steps:
                draw_marker(draw, px(step), py(float(value)), meta["color"], meta["marker"], 6)

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=base.GRAY, width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=base.GRAY, width=2)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_base()
    data: dict[str, dict] = {}
    for key, meta in RUNS.items():
        rows = base.parse_grpo(meta["path"])
        raw = [float(row["train_success"]) for row in rows]
        data[key] = {
            "steps": [int(row["step"]) for row in rows],
            "raw": raw,
            "ma5": base.trailing(raw, 5),
            "ma10": base.trailing(raw, 10),
            "fixed": [float(row["eval_success"]) if row["eval_success"] is not None else None for row in rows],
        }

    maximum = max(values["steps"][-1] for values in data.values())
    image = Image.new("RGB", (1760, 2260), base.LIGHT)
    draw = base.add_header(
        image,
        TITLE,
        SUBTITLE,
    )
    draw_shared_legend(draw, base, data)
    panel(draw, base, (45, 220, 1715, 695), data, "raw", "Per-step training-rollout success", 0.64, 1.005, maximum)
    panel(draw, base, (45, 725, 1715, 1200), data, "ma5", "Trailing 5-step mean", 0.68, 1.005, maximum)
    panel(draw, base, (45, 1230, 1715, 1705), data, "ma10", "Trailing 10-step mean", 0.70, 1.005, maximum)
    panel(draw, base, (45, 1735, 1715, 2210), data, "fixed", "Fixed-32 evaluation (recorded every 5 steps)", 0.68, 1.005, maximum)

    figure = OUT / "01_all_six_2gpu_grpo_raw_ma5_ma10_fixed32.png"
    image.save(figure, optimize=True)

    rows_out: list[dict[str, float | int | None]] = []
    for step in range(1, maximum + 1):
        row: dict[str, float | int | None] = {"step": step}
        for key in RUNS:
            for metric in ("raw", "ma5", "ma10", "fixed"):
                row[f"{key}_{metric}"] = dict(zip(data[key]["steps"], data[key][metric])).get(step)
        rows_out.append(row)
    with (OUT / "curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)

    summary: dict[str, object] = {"maximum_step": maximum, "runs": {}}
    for key in RUNS:
        raw = data[key]["raw"]
        fixed = [value for value in data[key]["fixed"] if value is not None]
        summary["runs"][key] = {
            "name": RUNS[key]["name"],
            "complete_step": data[key]["steps"][-1],
            "latest_raw_pct": raw[-1] * 100,
            "latest_ma5_pct": statistics.mean(raw[-5:]) * 100 if len(raw) >= 5 else None,
            "latest_ma10_pct": statistics.mean(raw[-10:]) * 100 if len(raw) >= 10 else None,
            "fixed32_successes": sum(round(value * 32) for value in fixed),
            "fixed32_episodes": len(fixed) * 32,
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figure": str(figure), "summary": summary}))


if __name__ == "__main__":
    main()
