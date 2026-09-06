from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "docs" / "rlinf-robotwin-pi0-dvac-telemetry"
OLD_CURVES = TOPIC / "evidence" / "rlt_success_bc_matched_width_stopped_final_20260827" / "curves.csv"
PURE02_05 = ROOT / "tmp" / "rlt_dvac_pure_live_refresh_20260828" / "raw"
PURE03_04 = ROOT / "tmp" / "rlt_dvac_pure03_pure04_live_refresh_20260829" / "raw"
PARSER = ROOT / "tmp" / "build_rlt_dvac_pure_live_g146_20260828.py"
OUT = Path(os.environ.get("RLT_SIX_OUT", str(TOPIC / "evidence" / "rlt_six_single_gpu_success_rows_g399_20260830")))

ORDER = ("clean", "old", "pure02", "pure03", "pure04", "pure05")
LABELS = {
    "clean": "Clean single-GPU RLT",
    "old": "Old DVAC-BC (executed target)",
    "pure02": "Pure02 (reference, strength 0.5)",
    "pure03": "Pure03 (reference, strength 1.0)",
    "pure04": "Pure04 (reference, strength 1.5)",
    "pure05": "Pure05 (reference, strength 2.0)",
}
COLORS = {
    "clean": "#0072B2",
    "old": "#CC79A7",
    "pure02": "#009E73",
    "pure03": "#6A3D9A",
    "pure04": "#E69F00",
    "pure05": "#D55E00",
}
PATTERNS = {
    "clean": None,
    "old": (9, 6),
    "pure02": (7, 5),
    "pure03": (6, 3),
    "pure04": (4, 3),
    "pure05": (3, 1),
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = "segoeuib.ttf" if bold else "segoeui.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / filename), size=size)


def rolling(rows: list[dict[str, float]], window: int) -> None:
    for index, row in enumerate(rows):
        values = [float(item["raw"]) for item in rows[max(0, index - window + 1) : index + 1]]
        row[f"ma{window}"] = sum(values) / len(values)


def load_series() -> dict[str, list[dict[str, float]]]:
    series: dict[str, list[dict[str, float]]] = {"clean": [], "old": []}
    with OLD_CURVES.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            step = int(float(row["step"]))
            series["clean"].append({"step": step, "raw": float(row["control_raw_pct"])})
            series["old"].append({"step": step, "raw": float(row["method_raw_pct"])})

    spec = importlib.util.spec_from_file_location("pure_parser", PARSER)
    parser = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(parser)
    series.update(
        {
            "pure02": parser.parse_metrics(PURE02_05 / "s0p5_metrics.log"),
            "pure03": parser.parse_metrics(PURE03_04 / "pure03_metrics.log"),
            "pure04": parser.parse_metrics(PURE03_04 / "pure04_metrics.log"),
            "pure05": parser.parse_metrics(PURE02_05 / "s2p0_metrics.log"),
        }
    )
    for rows in series.values():
        for window in (5, 10, 20):
            rolling(rows, window)
    return series


def patterned_line(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], name: str, width: int) -> None:
    pattern = PATTERNS[name]
    if pattern is None:
        draw.line(points, fill=COLORS[name], width=width)
        return
    period, on = pattern
    for index in range(len(points) - 1):
        if index % period < on:
            draw.line((points[index], points[index + 1]), fill=COLORS[name], width=width)


def render(series: dict[str, list[dict[str, float]]], max_step: int) -> Path:
    width, height = 2600, 3070
    image = Image.new("RGB", (width, height), "#F5F7FA")
    draw = ImageDraw.Draw(image)
    draw.text((90, 48), "Single-GPU RLT training-rollout success — six settings", fill="#172033", font=font(51, True))
    steps = " · ".join(f"{name.replace('pure', 'Pure').replace('clean', 'Clean').replace('old', 'Old')}: {int(series[name][-1]['step'])}" for name in ORDER)
    draw.text((90, 118), f"Latest real step — {steps}", fill="#52627A", font=font(23))
    draw.text((90, 160), "All rows use Step 1–480; Pure03/Pure04 stop at their current live step.", fill="#52627A", font=font(23))

    legend_positions = ((105, 250), (930, 250), (1740, 250), (105, 330), (930, 330), (1740, 330))
    for name, (x, y) in zip(ORDER, legend_positions):
        demo = [(x + offset * 8, y) for offset in range(14)]
        patterned_line(draw, demo, name, 8)
        draw.text((x + 130, y), LABELS[name], fill="#263547", font=font(22), anchor="lm")

    panels = (
        ((175, 570, 2460, 1010), "Per-step success", "raw"),
        ((175, 1240, 2460, 1680), "Trailing 5-step mean", "ma5"),
        ((175, 1910, 2460, 2350), "Trailing 10-step mean", "ma10"),
        ((175, 2580, 2460, 3020), "Trailing 20-step mean", "ma20"),
    )

    def point(step: int, value: float, box: tuple[int, int, int, int]) -> tuple[int, int]:
        left, top, right, bottom = box
        x = left + (step - 1) / (max_step - 1) * (right - left)
        y = bottom - value / 100 * (bottom - top)
        return round(x), round(y)

    for box, title, key in panels:
        left, top, right, bottom = box
        draw.rounded_rectangle((left - 82, top - 88, right + 45, bottom + 50), radius=20, fill="#FFFFFF", outline="#CFD9E5", width=2)
        draw.text((left - 35, top - 68), title, fill="#172033", font=font(34, True))
        for ytick in (0, 25, 50, 75, 100):
            y = point(1, ytick, box)[1]
            draw.line((left, y, right, y), fill="#D8E0E9", width=1)
            draw.text((left - 18, y), str(ytick), fill="#65758B", font=font(21), anchor="rm")
        for xtick in (1, 100, 200, 300, 400, 480):
            x = point(xtick, 0, box)[0]
            draw.line((x, bottom, x, bottom + 8), fill="#65758B", width=1)
            draw.text((x, bottom + 18), str(xtick), fill="#65758B", font=font(20), anchor="ma")
        for name in ORDER:
            points = [point(int(row["step"]), float(row[key]), box) for row in series[name]]
            patterned_line(draw, points, name, 2 if key == "raw" else 5)
            x, y = points[-1]
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="#FFFFFF", outline=COLORS[name], width=4)

    path = OUT / "RLT_SIX_SINGLE_GPU_SUCCESS_ROWS_STEP1_480.png"
    image.save(path)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    series = load_series()
    max_step = max(int(rows[-1]["step"]) for rows in series.values())
    lookup = {name: {int(row["step"]): row for row in rows} for name, rows in series.items()}
    fields = ["step"] + [f"{name}_{metric}_pct" for name in ORDER for metric in ("raw", "ma5", "ma10", "ma20")]
    with (OUT / "success_curves.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step in range(1, max_step + 1):
            row: dict[str, object] = {"step": step}
            for name in ORDER:
                source = lookup[name].get(step)
                for metric in ("raw", "ma5", "ma10", "ma20"):
                    row[f"{name}_{metric}_pct"] = "" if source is None else source[metric]
            writer.writerow(row)
    summary = {"max_step": max_step, "runs": {}}
    for name in ORDER:
        rows = series[name]
        latest = rows[-1]
        summary["runs"][name] = {
            "label": LABELS[name],
            "latest_step": int(latest["step"]),
            "cumulative_mean_pct": sum(float(row["raw"]) for row in rows) / len(rows),
            "latest_raw_pct": float(latest["raw"]),
            "latest_ma5_pct": float(latest["ma5"]),
            "latest_ma10_pct": float(latest["ma10"]),
            "latest_ma20_pct": float(latest["ma20"]),
        }
    plot = render(series, max_step)
    summary["plot"] = plot.name
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
