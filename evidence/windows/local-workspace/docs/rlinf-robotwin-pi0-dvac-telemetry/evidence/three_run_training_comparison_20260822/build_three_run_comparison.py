from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
BASELINE_JSON = ROOT / "audits/20260717-084926-grpo-current/analysis.json"
V1_ZIP = ROOT / "exports/idea2_dvac_v1_formal_stop_g54_20260821.zip"
V1_MEMBER = (
    "idea2_dvac_v1_formal_stop_g54_20260821/analysis/TRAIN_METRICS.csv"
)
V2_CSV = (
    ROOT
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/"
    "r_only_formal_live_g23_20260822/analysis/TRAIN_METRICS_G23.csv"
)


def numeric(value: str | float | int | None) -> float:
    if value in (None, ""):
        return float("nan")
    return float(value)


def read_baseline() -> list[dict[str, float]]:
    payload = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    rows = []
    for row in payload["metrics"]:
        rows.append(
            {
                "step": float(row["step"]),
                "success": 100.0 * numeric(row.get("success_once")),
                "kl": numeric(row.get("actor/approx_kl")),
                "clip": 100.0 * numeric(row.get("actor/clip_fraction")),
                "grad": numeric(row.get("actor/grad_norm")),
                "step_min": numeric(row.get("step_time_s")) / 60.0,
            }
        )
    return rows


def normalize_csv_rows(raw_rows: list[dict[str, str]]) -> list[dict[str, float]]:
    rows = []
    for row in raw_rows:
        rows.append(
            {
                "step": numeric(row.get("global_step")),
                "success": 100.0 * numeric(row.get("success_once")),
                "kl": numeric(row.get("actor/approx_kl")),
                "clip": 100.0 * numeric(row.get("actor/clip_fraction")),
                "grad": numeric(row.get("actor/grad_norm")),
                "step_min": numeric(row.get("step_time_s")) / 60.0,
            }
        )
    return rows


def read_v1() -> list[dict[str, float]]:
    with zipfile.ZipFile(V1_ZIP) as archive:
        text = archive.read(V1_MEMBER).decode("utf-8-sig")
    return normalize_csv_rows(list(csv.DictReader(text.splitlines())))


def read_v2() -> list[dict[str, float]]:
    with V2_CSV.open(encoding="utf-8-sig", newline="") as handle:
        return normalize_csv_rows(list(csv.DictReader(handle)))


def rolling(values: np.ndarray, window: int = 5) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    for index in range(len(values)):
        result[index] = np.mean(values[max(0, index - window + 1) : index + 1])
    return result


def vector(rows: list[dict[str, float]], key: str) -> np.ndarray:
    return np.asarray([row[key] for row in rows], dtype=float)


RUNS = {
    "Original GRPO (100 steps)": (read_baseline(), "#2563EB"),
    "DVAC v1 global-z (54 steps)": (read_v1(), "#F59E0B"),
    "DVAC v2 R-only (g23 snapshot)": (read_v2(), "#7C3AED"),
}


FONT_REGULAR = Path(r"C:\Windows\Fonts\segoeui.ttf")
FONT_SEMIBOLD = Path(r"C:\Windows\Fonts\seguisb.ttf")
INK = "#17233D"
MUTED = "#5E6C84"
GRID = "#D9E1EA"
WHITE = "#FFFFFF"


def font(size: int, semibold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_SEMIBOLD if semibold else FONT_REGULAR), size=size)


def draw_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    series: list[tuple[str, np.ndarray, np.ndarray, str]],
    ylabel: str,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float] | None = None,
    reference: float | None = None,
    legend: bool = True,
) -> None:
    left, top, right, bottom = box
    draw.text((left, top), title, fill=INK, font=font(27, True))
    plot_left, plot_right = left + 92, right - 24
    plot_top, plot_bottom = top + 58, bottom - 48
    values = np.concatenate([ys[np.isfinite(ys)] for _, _, ys, _ in series])
    if y_limits is None:
        low, high = float(np.min(values)), float(np.max(values))
        pad = 0.12 * (high - low) if high > low else max(abs(high) * 0.1, 1.0)
        low, high = low - pad, high + pad
    else:
        low, high = y_limits
    x_low, x_high = x_limits

    def px(value: float) -> float:
        return plot_left + (value - x_low) / (x_high - x_low) * (plot_right - plot_left)

    def py(value: float) -> float:
        return plot_bottom - (value - low) / (high - low) * (plot_bottom - plot_top)

    for tick in range(5):
        value = low + (high - low) * tick / 4
        y = py(value)
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=2)
        label = f"{value:.3f}" if abs(value) < 1 else f"{value:.1f}"
        width = draw.textlength(label, font=font(17))
        draw.text((plot_left - width - 12, y - 10), label, fill=MUTED, font=font(17))
    for tick in range(5):
        value = x_low + (x_high - x_low) * tick / 4
        x = px(value)
        label = str(int(round(value)))
        width = draw.textlength(label, font=font(17))
        draw.text((x - width / 2, plot_bottom + 10), label, fill=MUTED, font=font(17))
    if reference is not None and low <= reference <= high:
        y = py(reference)
        for x in range(int(plot_left), int(plot_right), 18):
            draw.line((x, y, min(x + 9, plot_right), y), fill=MUTED, width=2)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=INK, width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=INK, width=2)
    draw.text((left, plot_top), ylabel, fill=MUTED, font=font(17))

    legend_x, legend_y = plot_left, plot_top + 5
    for label, xs, ys, color in series:
        points = [
            (px(float(x)), py(float(y)))
            for x, y in zip(xs, ys)
            if np.isfinite(x) and np.isfinite(y)
        ]
        if len(points) > 1:
            draw.line(points, fill=color, width=4, joint="curve")
        if points:
            x, y = points[-1]
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
        if legend:
            draw.line((legend_x, legend_y + 11, legend_x + 28, legend_y + 11), fill=color, width=5)
            draw.text((legend_x + 36, legend_y), label, fill=INK, font=font(17))
            legend_x += int(draw.textlength(label, font=font(17))) + 78


success_image = Image.new("RGB", (1800, 1500), WHITE)
success_draw = ImageDraw.Draw(success_image)
success_draw.text((70, 45), "Three GRPO training runs: rollout success", fill=INK, font=font(43, True))
success_draw.text(
    (72, 105),
    "On-policy training success, not fixed-ID evaluation. Each curve ends at its latest complete step.",
    fill=MUTED,
    font=font(22),
)
raw_series = []
rolling_series = []
for label, (rows, color) in RUNS.items():
    steps = vector(rows, "step")
    success = vector(rows, "success")
    raw_series.append((label, steps, success, color))
    rolling_series.append((label, steps, rolling(success, 5), color))
draw_chart(success_draw, (55, 180, 1745, 585), "A. Raw success at each global step", raw_series, "%", (1, 100), (65, 102))
draw_chart(success_draw, (55, 620, 1745, 1025), "B. Five-step rolling mean", rolling_series, "%", (1, 100), (72, 101))
baseline_rows = RUNS["Original GRPO (100 steps)"][0]
baseline_roll = rolling(vector(baseline_rows, "success"), 5)
delta_series = []
for label in ("DVAC v1 global-z (54 steps)", "DVAC v2 R-only (g23 snapshot)"):
    rows, color = RUNS[label]
    steps = vector(rows, "step").astype(int)
    delta_series.append((f"{label} minus GRPO", steps, rolling(vector(rows, "success"), 5) - baseline_roll[steps - 1], color))
draw_chart(success_draw, (55, 1060, 1745, 1460), "C. Rolling-5 difference from original GRPO", delta_series, "pp", (1, 55), reference=0)
success_image.save(HERE / "THREE_RUN_SUCCESS_COMPARISON_G23.png", optimize=True)


opt_image = Image.new("RGB", (1800, 1120), WHITE)
opt_draw = ImageDraw.Draw(opt_image)
opt_draw.text((70, 45), "Three GRPO runs: optimization-side metrics", fill=INK, font=font(43, True))
metric_specs = [
    ("kl", "A. Approx KL", "fraction"),
    ("clip", "B. PPO joint-query clip fraction", "% queries"),
    ("grad", "C. Pre-global-clip gradient norm", "norm"),
    ("step_min", "D. Global-step wall time", "minutes"),
]
boxes = [(45, 135, 885, 590), (915, 135, 1755, 590), (45, 625, 885, 1080), (915, 625, 1755, 1080)]
for box, (key, title, ylabel) in zip(boxes, metric_specs):
    series = [(label, vector(rows, "step"), vector(rows, key), color) for label, (rows, color) in RUNS.items()]
    draw_chart(opt_draw, box, title, series, ylabel, (1, 100), legend=title.startswith("A."))
opt_image.save(HERE / "THREE_RUN_OPTIMIZATION_COMPARISON_G23.png", optimize=True)


all_steps = sorted({int(row["step"]) for rows, _ in RUNS.values() for row in rows})
with (HERE / "THREE_RUN_METRICS_G23.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    fieldnames = ["run", "step", "success_pct", "success_roll5_pct", "approx_kl", "clip_pct", "grad_norm", "step_min"]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for label, (rows, _) in RUNS.items():
        success = vector(rows, "success")
        roll5 = rolling(success, 5)
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "run": label,
                    "step": int(row["step"]),
                    "success_pct": row["success"],
                    "success_roll5_pct": roll5[index],
                    "approx_kl": row["kl"],
                    "clip_pct": row["clip"],
                    "grad_norm": row["grad"],
                    "step_min": row["step_min"],
                }
            )

summary = {}
for label, (rows, _) in RUNS.items():
    success = vector(rows, "success")
    summary[label] = {
        "complete_steps": len(rows),
        "latest_step": int(rows[-1]["step"]),
        "success_mean_pct": float(np.mean(success)),
        "success_latest_pct": float(success[-1]),
        "success_roll5_latest_pct": float(rolling(success, 5)[-1]),
        "mean_approx_kl": float(np.mean(vector(rows, "kl"))),
        "mean_clip_pct": float(np.mean(vector(rows, "clip"))),
        "mean_grad_norm": float(np.mean(vector(rows, "grad"))),
        "mean_step_min": float(np.mean(vector(rows, "step_min"))),
    }
(HERE / "THREE_RUN_SUMMARY_G23.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)

print(json.dumps(summary, indent=2, ensure_ascii=False))
