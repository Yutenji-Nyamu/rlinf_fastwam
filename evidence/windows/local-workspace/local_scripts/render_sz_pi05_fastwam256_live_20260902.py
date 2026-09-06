"""Render a compact high-contrast live success chart from the read-only snapshot."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"C:\Users\86136\Documents\rl")
SRC = (
    ROOT
    / "docs"
    / "rlinf-shenzhen-experiment-expansion"
    / "evidence"
    / "current-pair-live-20260902-2046"
)
OUT = SRC / "figures"
COLORS = {"pi05_control": "#006D77", "fastwam256": "#D55E00"}
LABELS = {"pi05_control": "pi0.5 GRPO", "fastwam256": "Fast-WAM GRPO (256 traj)"}
BG, PANEL, GRID, TEXT, MUTED = "#F5F7FA", "#FFFFFF", "#DCE3EC", "#172033", "#667085"


def font(size: int, bold: bool = False):
    filename = "seguisb.ttf" if bold else "segoeui.ttf"
    return ImageFont.truetype(str(Path(r"C:\Windows\Fonts") / filename), size)


F_TITLE = font(34, True)
F_SUB = font(20)
F_PANEL = font(24, True)
F_LABEL = font(17)
F_LEGEND = font(18, True)


def rolling(points: list[dict], window: int) -> list[tuple[int, float]]:
    values = [float(p["value"]) for p in points]
    return [
        (int(points[i]["step"]), sum(values[max(0, i - window + 1):i + 1]) / min(window, i + 1))
        for i in range(len(points))
    ]


def raw(points: list[dict]) -> list[tuple[int, float]]:
    return [(int(p["step"]), float(p["value"])) for p in points]


def draw_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str,
               series: dict[str, list[tuple[int, float]]], xmax: int) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=18, fill=PANEL, outline="#D2DAE5", width=2)
    draw.text((x0 + 26, y0 + 18), title, fill=TEXT, font=F_PANEL)
    left, top, right, bottom = x0 + 96, y0 + 76, x1 - 30, y1 - 52
    ymin, ymax = 0.20, 1.0

    def px(x: float) -> float:
        return left + (x - 1) / max(1, xmax - 1) * (right - left)

    def py(y: float) -> float:
        return bottom - (y - ymin) / (ymax - ymin) * (bottom - top)

    for i in range(5):
        y = ymin + i * 0.20
        yy = py(y)
        draw.line((left, yy, right, yy), fill=GRID, width=2)
        draw.text((left - 12, yy), f"{y:.0%}", fill=MUTED, font=F_LABEL, anchor="rm")
    for i in range(6):
        x = 1 + (xmax - 1) * i / 5
        xx = px(x)
        draw.line((xx, top, xx, bottom), fill="#EEF1F5", width=1)
        draw.text((xx, bottom + 9), f"{x:.0f}", fill=MUTED, font=F_LABEL, anchor="ma")
    draw.line((left, bottom, right, bottom), fill="#778396", width=2)
    draw.line((left, top, left, bottom), fill="#778396", width=2)

    legend_x = x0 + 650
    for key in ("pi05_control", "fastwam256"):
        color = COLORS[key]
        draw.line((legend_x, y0 + 35, legend_x + 44, y0 + 35), fill=color, width=6)
        draw.text((legend_x + 54, y0 + 35), LABELS[key], fill=color, font=F_LEGEND, anchor="lm")
        legend_x += 360
        points = [(px(x), py(y)) for x, y in series[key]]
        if len(points) >= 2:
            draw.line(points, fill=color, width=5, joint="curve")
        for xx, yy in points:
            if key == "pi05_control":
                draw.ellipse((xx - 4, yy - 4, xx + 4, yy + 4), fill=color)
            else:
                draw.rectangle((xx - 6, yy - 6, xx + 6, yy + 6), fill=color)
    draw.text(((left + right) / 2, y1 - 17), "completed global step", fill=MUTED, font=F_LABEL, anchor="mm")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads((SRC / "metrics.json").read_text(encoding="utf-8"))
    trains = {key: run["series"]["env/success_once"] for key, run in data.items()}
    xmax = max(int(p["step"]) for points in trains.values() for p in points)

    image = Image.new("RGB", (1600, 1330), BG)
    draw = ImageDraw.Draw(image)
    draw.text((48, 28), "SZ live training — success through 2026-09-02 20:49 CST", fill=TEXT, font=F_TITLE)
    draw.text((48, 78), "Deep teal: pi0.5 Control. Vermilion: Fast-WAM256. Early MA uses available completed steps.", fill=MUTED, font=F_SUB)
    boxes = [(38, 122, 1562, 500), (38, 520, 1562, 898), (38, 918, 1562, 1296)]
    draw_panel(draw, boxes[0], "Per-step training-rollout success", {k: raw(v) for k, v in trains.items()}, xmax)
    draw_panel(draw, boxes[1], "Trailing 5-step mean", {k: rolling(v, 5) for k, v in trains.items()}, xmax)
    draw_panel(draw, boxes[2], "Trailing 10-step mean", {k: rolling(v, 10) for k, v in trains.items()}, xmax)
    image.save(OUT / "01_pi05_fastwam256_success_raw_ma5_ma10.png", quality=96)

    rows = []
    summary = {}
    for key, run in data.items():
        train = run["series"]["env/success_once"]
        ma5 = rolling(train, 5)
        ma10 = rolling(train, 10)
        times = [float(x["value"]) for x in run["series"]["time/step"]]
        evals = run["series"].get("eval/success_once", [])
        summary[key] = {
            "completed_step": int(train[-1]["step"]),
            "latest_success_pct": float(train[-1]["value"]) * 100,
            "ma5_pct": ma5[-1][1] * 100,
            "ma10_pct": ma10[-1][1] * 100,
            "latest_fixed_eval": evals[-1] if evals else None,
            "latest_kl": float(run["series"]["train/actor/approx_kl"][-1]["value"]),
            "latest_clip": float(run["series"]["train/actor/clip_fraction"][-1]["value"]),
            "latest_grad": float(run["series"]["train/actor/grad_norm"][-1]["value"]),
            "median_step_seconds": median(times[-10:]),
        }
        lookup5 = dict(ma5)
        lookup10 = dict(ma10)
        for point in train:
            step = int(point["step"])
            rows.append((key, step, float(point["value"]) * 100, lookup5[step] * 100, lookup10[step] * 100))
    (SRC / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (SRC / "success_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["experiment", "step", "success_pct", "ma5_pct", "ma10_pct"])
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
