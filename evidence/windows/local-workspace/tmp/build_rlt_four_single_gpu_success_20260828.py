from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "docs" / "rlinf-robotwin-pi0-dvac-telemetry"
OLD = TOPIC / "evidence" / "rlt_success_bc_matched_width_stopped_final_20260827" / "curves.csv"
PURE_RAW = ROOT / "tmp" / "rlt_dvac_pure_live_refresh_20260828" / "raw"
PARSER_PATH = ROOT / "tmp" / "build_rlt_dvac_pure_live_g146_20260828.py"


def load_pure_parser():
    spec = importlib.util.spec_from_file_location("pure_live_builder", PARSER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.parse_metrics


def load_old() -> dict[str, list[dict[str, float]]]:
    result = {"rlt": [], "dvac_bc": []}
    with OLD.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            step = int(float(raw["step"]))
            for name, prefix in (("rlt", "control"), ("dvac_bc", "method")):
                result[name].append(
                    {
                        "step": step,
                        "raw": float(raw[f"{prefix}_raw_pct"]),
                        "ma5": float(raw[f"{prefix}_ma5_pct"]),
                        "ma10": float(raw[f"{prefix}_ma10_pct"]),
                    }
                )
    return result


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = "segoeuib.ttf" if bold else "segoeui.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / filename), size=size)


def dashed_line(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], color: str, width: int) -> None:
    for index in range(len(points) - 1):
        if (index // 3) % 2 == 0:
            draw.line((points[index], points[index + 1]), fill=color, width=width)


def main() -> None:
    parse_metrics = load_pure_parser()
    series = load_old()
    series["pure_s0p5"] = parse_metrics(PURE_RAW / "s0p5_metrics.log")
    series["pure_s2p0"] = parse_metrics(PURE_RAW / "s2p0_metrics.log")
    common = min(int(rows[-1]["step"]) for rows in series.values())
    series = {
        name: [row for row in rows if int(row["step"]) <= common]
        for name, rows in series.items()
    }

    out = TOPIC / "evidence" / f"rlt_four_single_gpu_success_live_g{common}_20260828"
    out.mkdir(parents=True, exist_ok=True)

    labels = {
        "rlt": "RLT clean",
        "dvac_bc": "RLT-DVAC-BC (previous)",
        "pure_s0p5": "Pure [0,2] / s0p5",
        "pure_s2p0": "Pure [0,5] / s2p0",
    }
    colors = {
        "rlt": "#0072B2",
        "dvac_bc": "#CC79A7",
        "pure_s0p5": "#009E73",
        "pure_s2p0": "#D55E00",
    }

    fields = ["step"] + [f"{name}_{key}_pct" for name in series for key in ("raw", "ma5", "ma10")]
    by_run = {name: {int(row["step"]): row for row in rows} for name, rows in series.items()}
    with (out / "success_curves.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step in range(1, common + 1):
            row: dict[str, object] = {"step": step}
            for name, rows in by_run.items():
                for key in ("raw", "ma5", "ma10"):
                    row[f"{name}_{key}_pct"] = rows[step][key]
            writer.writerow(row)

    summary = {
        "common_step": common,
        "runs": {
            name: {
                "latest_raw_pct": float(rows[-1]["raw"]),
                "latest_ma5_pct": float(rows[-1]["ma5"]),
                "latest_ma10_pct": float(rows[-1]["ma10"]),
                "cumulative_mean_pct": sum(float(row["raw"]) for row in rows) / len(rows),
            }
            for name, rows in series.items()
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    width, height = 1900, 1530
    image = Image.new("RGB", (width, height), "#F6F8FB")
    draw = ImageDraw.Draw(image)
    draw.text((70, 44), f"Four matched single-GPU RLT runs through Step {common}", fill="#172033", font=font(40, True))
    draw.text(
        (70, 99),
        "Training-rollout success; all four curves are truncated to the same runner-step range.",
        fill="#52627A",
        font=font(22),
    )

    legend_positions = [(85, 157), (510, 157), (1040, 157), (1450, 157)]
    for (name, label), (x, y) in zip(labels.items(), legend_positions):
        draw.line((x, y, x + 52, y), fill=colors[name], width=7)
        draw.text((x + 67, y), label, fill="#334155", font=font(19), anchor="lm")

    panels = [
        ((110, 300, 1790, 620), "Per-step success", "raw"),
        ((110, 765, 1790, 1085), "Trailing 5-step mean", "ma5"),
        ((110, 1230, 1790, 1500), "Trailing 10-step mean", "ma10"),
    ]

    def point(step: int, value: float, box: tuple[int, int, int, int]) -> tuple[int, int]:
        left, top, right, bottom = box
        x = left + (step - 1) / max(1, common - 1) * (right - left)
        y = bottom - value / 100.0 * (bottom - top)
        return round(x), round(y)

    for box, title, key in panels:
        left, top, right, bottom = box
        draw.rounded_rectangle((left - 55, top - 68, right + 35, bottom + 28), 16, fill="#FFFFFF", outline="#D7E0EA", width=2)
        draw.text((left - 20, top - 53), title, fill="#172033", font=font(27, True))
        for tick in (0, 25, 50, 75, 100):
            y = point(1, tick, box)[1]
            draw.line((left, y, right, y), fill="#D9E1EA", width=1)
            draw.text((left - 15, y), str(tick), fill="#64748B", font=font(17), anchor="rm")
        xticks = list(range(25, common + 1, 25))
        if common not in xticks:
            xticks.append(common)
        for tick in xticks:
            x = point(tick, 0, box)[0]
            draw.line((x, bottom, x, bottom + 6), fill="#64748B", width=1)
            draw.text((x, bottom + 12), str(tick), fill="#64748B", font=font(16), anchor="ma")
        for name, rows in series.items():
            points = [point(int(row["step"]), float(row[key]), box) for row in rows]
            if name in ("rlt", "dvac_bc"):
                dashed_line(draw, points, colors[name], 4 if key != "raw" else 3)
            else:
                draw.line(points, fill=colors[name], width=4 if key != "raw" else 3)

    plot = out / f"RLT_FOUR_SINGLE_GPU_SUCCESS_THROUGH_G{common}.png"
    image.save(plot)
    print(json.dumps({"out": str(out), "plot": str(plot), **summary}, indent=2))


if __name__ == "__main__":
    main()
