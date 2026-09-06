from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "docs" / "rlinf-robotwin-pi0-dvac-telemetry"
OLD_CURVES = (
    TOPIC
    / "evidence"
    / "rlt_success_bc_matched_width_stopped_final_20260827"
    / "curves.csv"
)
PURE_RAW = ROOT / "tmp" / "rlt_dvac_pure_live_refresh_20260828" / "raw"
PURE_PARSER = ROOT / "tmp" / "build_rlt_dvac_pure_live_g146_20260828.py"
OUT = TOPIC / "evidence" / "rlt_four_single_gpu_success_full_axis_20260828"

RUN_ORDER = ("clean", "success_dvac_bc", "pure02", "pure05")
LABELS = {
    "clean": "Clean single-GPU RLT",
    "success_dvac_bc": "Old DVAC-BC (executed target)",
    "pure02": "Pure02 (reference target, s0.5)",
    "pure05": "Pure05 (reference target, s2.0)",
}
COLORS = {
    "clean": "#0057B8",
    "success_dvac_bc": "#C2185B",
    "pure02": "#008B5A",
    "pure05": "#E65100",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = "segoeuib.ttf" if bold else "segoeui.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / filename), size=size)


def rolling(rows: list[dict[str, float]], window: int) -> None:
    for index, row in enumerate(rows):
        values = [
            float(item["raw"])
            for item in rows[max(0, index - window + 1) : index + 1]
        ]
        row[f"ma{window}"] = sum(values) / len(values)


def load_historical() -> dict[str, list[dict[str, float]]]:
    series = {"clean": [], "success_dvac_bc": []}
    with OLD_CURVES.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            step = int(float(row["step"]))
            series["clean"].append(
                {"step": step, "raw": float(row["control_raw_pct"])}
            )
            series["success_dvac_bc"].append(
                {"step": step, "raw": float(row["method_raw_pct"])}
            )
    return series


def load_pure() -> dict[str, list[dict[str, float]]]:
    spec = importlib.util.spec_from_file_location("pure_parser", PURE_PARSER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return {
        "pure02": module.parse_metrics(PURE_RAW / "s0p5_metrics.log"),
        "pure05": module.parse_metrics(PURE_RAW / "s2p0_metrics.log"),
    }


def dash_line(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    color: str,
    width: int,
    period: int,
    on: int,
) -> None:
    for index in range(len(points) - 1):
        if index % period < on:
            draw.line((points[index], points[index + 1]), fill=color, width=width)


def render(series: dict[str, list[dict[str, float]]], max_step: int) -> Path:
    width, height = 2040, 1700
    canvas = Image.new("RGB", (width, height), "#F5F7FA")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (70, 42),
        "Single-GPU RLT training-rollout success",
        fill="#172033",
        font=font(43, True),
    )
    lengths = " · ".join(f"{LABELS[name]}: Step {int(series[name][-1]['step'])}" for name in RUN_ORDER)
    draw.text((70, 100), lengths, fill="#52627A", font=font(19))
    draw.text(
        (70, 135),
        f"All panels use the longest real axis, Step 1–{max_step}; shorter runs stop at their actual latest step.",
        fill="#52627A",
        font=font(19),
    )

    legend_xy = ((82, 194), (550, 194), (1115, 194), (1560, 194))
    for name, (x, y) in zip(RUN_ORDER, legend_xy):
        draw.line((x, y, x + 58, y), fill=COLORS[name], width=9)
        draw.text((x + 72, y), LABELS[name], fill="#263547", font=font(18), anchor="lm")

    panels = (
        ((115, 330, 960, 870), "Per-step success", "raw"),
        ((1100, 330, 1945, 870), "Trailing 5-step mean", "ma5"),
        ((115, 1080, 960, 1570), "Trailing 10-step mean", "ma10"),
        ((1100, 1080, 1945, 1570), "Trailing 20-step mean", "ma20"),
    )

    def point(step: int, value: float, box: tuple[int, int, int, int]) -> tuple[int, int]:
        left, top, right, bottom = box
        x = left + (step - 1) / max(1, max_step - 1) * (right - left)
        y = bottom - value / 100.0 * (bottom - top)
        return round(x), round(y)

    for box, title, key in panels:
        left, top, right, bottom = box
        draw.rounded_rectangle(
            (left - 62, top - 72, right + 35, bottom + 48),
            radius=18,
            fill="#FFFFFF",
            outline="#CFD9E5",
            width=2,
        )
        draw.text((left - 25, top - 56), title, fill="#172033", font=font(29, True))
        for ytick in (0, 25, 50, 75, 100):
            y = point(1, ytick, box)[1]
            draw.line((left, y, right, y), fill="#D8E0E9", width=1)
            draw.text((left - 15, y), str(ytick), fill="#65758B", font=font(17), anchor="rm")
        for xtick in (1, 100, 200, 300, 400, max_step):
            x = point(xtick, 0, box)[0]
            draw.line((x, bottom, x, bottom + 7), fill="#65758B", width=1)
            draw.text((x, bottom + 14), str(xtick), fill="#65758B", font=font(16), anchor="ma")

        for index, name in enumerate(RUN_ORDER):
            points = [point(int(row["step"]), float(row[key]), box) for row in series[name]]
            line_width = 2 if key == "raw" else 4
            if index == 0:
                draw.line(points, fill=COLORS[name], width=line_width)
            elif index == 1:
                dash_line(draw, points, COLORS[name], line_width, period=6, on=4)
            elif index == 2:
                dash_line(draw, points, COLORS[name], line_width, period=4, on=3)
            else:
                dash_line(draw, points, COLORS[name], line_width, period=3, on=1)
            x, y = points[-1]
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="#FFFFFF", outline=COLORS[name], width=4)

    output = OUT / f"RLT_FOUR_SINGLE_GPU_SUCCESS_FULL_AXIS_G{max_step}.png"
    canvas.save(output)
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    series = load_historical()
    series.update(load_pure())
    for rows in series.values():
        for window in (5, 10, 20):
            rolling(rows, window)

    max_step = max(int(rows[-1]["step"]) for rows in series.values())
    fieldnames = ["step"] + [
        f"{name}_{metric}_pct"
        for name in RUN_ORDER
        for metric in ("raw", "ma5", "ma10", "ma20")
    ]
    lookup = {
        name: {int(row["step"]): row for row in rows}
        for name, rows in series.items()
    }
    with (OUT / "success_curves.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for step in range(1, max_step + 1):
            output_row: dict[str, object] = {"step": step}
            for name in RUN_ORDER:
                source = lookup[name].get(step)
                for metric in ("raw", "ma5", "ma10", "ma20"):
                    output_row[f"{name}_{metric}_pct"] = "" if source is None else source[metric]
            writer.writerow(output_row)

    common_step = min(int(rows[-1]["step"]) for rows in series.values())
    summary = {"max_step": max_step, "common_step": common_step, "runs": {}}
    for name in RUN_ORDER:
        rows = series[name]
        summary["runs"][name] = {
            "label": LABELS[name],
            "latest_step": int(rows[-1]["step"]),
            "latest_raw_pct": float(rows[-1]["raw"]),
            "latest_ma5_pct": float(rows[-1]["ma5"]),
            "latest_ma10_pct": float(rows[-1]["ma10"]),
            "latest_ma20_pct": float(rows[-1]["ma20"]),
            "cumulative_mean_pct": sum(float(row["raw"]) for row in rows) / len(rows),
            "at_common_step": {
                metric: float(lookup[name][common_step][metric])
                for metric in ("raw", "ma5", "ma10", "ma20")
            },
        }
    plot = render(series, max_step)
    summary["plot"] = plot.name
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
