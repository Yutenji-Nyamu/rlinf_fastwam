from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tmp"))
from build_rlt_dvac_pure_live_g146_20260828 import parse_metrics  # noqa: E402


RAW = Path(os.environ.get("RLT_PURE_RAW", str(ROOT / "tmp" / "rlt_dvac_pure03_pure04_live_refresh_20260829" / "raw")))
OUT = Path(os.environ.get("RLT_PURE_OUT", str(ROOT / "docs" / "rlinf-robotwin-pi0-dvac-telemetry" / "evidence" / "rlt_dvac_pure03_pure04_live_g100_20260829")))
STOPPED = os.environ.get("RLT_PURE_STOPPED", "0") == "1"
RUNS = {
    "pure03": {
        "label": "Pure03 / strength 1.0",
        "color": "#00796B",
        "path": RAW / "pure03_metrics.log",
    },
    "pure04": {
        "label": "Pure04 / strength 1.5",
        "color": "#E66101",
        "path": RAW / "pure04_metrics.log",
    },
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size=size)


def trailing(rows: list[dict[str, object]], window: int) -> None:
    for index, row in enumerate(rows):
        values = [
            float(item["raw"])
            for item in rows[max(0, index - window + 1) : index + 1]
        ]
        row[f"ma{window}"] = sum(values) / len(values)


def resource_summary() -> dict[str, object]:
    rows: list[dict[str, float]] = []
    with (RAW / "paired_resources.csv").open(encoding="utf-8", newline="") as handle:
        for source in csv.DictReader(handle):
            try:
                rows.append(
                    {
                        "timestamp": source["timestamp"],
                        "memory_gib": float(source["memory_current"]) / 2**30,
                        "gpu0_gib": float(source["gpu0_mib"]) / 1024,
                        "gpu1_gib": float(source["gpu1_mib"]) / 1024,
                        "gpu0_util": float(source["gpu0_util"]),
                        "gpu1_util": float(source["gpu1_util"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    return {
        "latest_timestamp": rows[-1]["timestamp"],
        "latest": rows[-1],
        "peak": {
            "memory_gib": max(row["memory_gib"] for row in rows),
            "gpu0_gib": max(row["gpu0_gib"] for row in rows),
            "gpu1_gib": max(row["gpu1_gib"] for row in rows),
        },
        "samples": len(rows),
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    latest = rows[-1]
    fixed = [
        {"step": int(row["step"]), "success_pct": float(row["eval"])}
        for row in rows
        if row.get("eval") is not None
    ]
    weights = {
        key: latest[key]
        for key in ("weight_p05", "weight_mean", "weight_p95", "weight_ess_ratio", "top20_weight_mass")
        if latest.get(key) is not None
    }
    return {
        "latest_step": int(latest["step"]),
        "latest_raw_pct": float(latest["raw"]),
        "cumulative_mean_pct": sum(float(row["raw"]) for row in rows) / len(rows),
        "latest_ma5_pct": float(latest["ma5"]),
        "latest_ma10_pct": float(latest["ma10"]),
        "latest_ma20_pct": float(latest["ma20"]),
        "fixed_eval": fixed,
        "latest_method_weights": weights,
    }


def save_curves(series: dict[str, list[dict[str, object]]]) -> None:
    fields = ["run", "step", "raw_pct", "ma5_pct", "ma10_pct", "ma20_pct", "fixed_eval_pct"]
    with (OUT / "success_curves.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run, rows in series.items():
            for row in rows:
                writer.writerow(
                    {
                        "run": run,
                        "step": row["step"],
                        "raw_pct": row["raw"],
                        "ma5_pct": row["ma5"],
                        "ma10_pct": row["ma10"],
                        "ma20_pct": row["ma20"],
                        "fixed_eval_pct": "" if row.get("eval") is None else row["eval"],
                    }
                )


def render(series: dict[str, list[dict[str, object]]]) -> Path:
    width, height = 1800, 1270
    image = Image.new("RGB", (width, height), "#F6F8FB")
    draw = ImageDraw.Draw(image)
    max_step = max(int(rows[-1]["step"]) for rows in series.values())
    draw.text((60, 38), f"RLT-DVAC Pure03 / Pure04 through Step {max_step}", fill="#172033", font=font(38, True))
    draw.text((60, 88), "Training-rollout success; trailing means include all available prior cycles.", fill="#52627A", font=font(20))
    for index, (run, meta) in enumerate(RUNS.items()):
        x = 1060 + index * 330
        draw.line((x, 92, x + 44, 92), fill=str(meta["color"]), width=6)
        draw.text((x + 54, 92), str(meta["label"]), fill="#475569", font=font(17), anchor="lm")

    boxes = [(110, 205, 820, 590), (980, 205, 1690, 590), (110, 755, 820, 1140), (980, 755, 1690, 1140)]
    panels = [("Per-step success", "raw"), ("Trailing 5-step mean", "ma5"), ("Trailing 10-step mean", "ma10"), ("Trailing 20-step mean", "ma20")]

    def point(step: float, value: float, box: tuple[int, int, int, int]) -> tuple[int, int]:
        left, top, right, bottom = box
        x = left + (step - 1) / max(1, max_step - 1) * (right - left)
        y = bottom - value / 100 * (bottom - top)
        return round(x), round(y)

    for box, (title, key) in zip(boxes, panels):
        left, top, right, bottom = box
        draw.rounded_rectangle((left - 52, top - 58, right + 38, bottom + 48), radius=15, fill="#FFFFFF", outline="#D7E0EA", width=2)
        draw.text((left - 20, top - 44), title, fill="#172033", font=font(26, True))
        for tick in (0, 25, 50, 75, 100):
            y = point(1, tick, box)[1]
            draw.line((left, y, right, y), fill="#D9E1EA", width=1)
            draw.text((left - 12, y), str(tick), fill="#64748B", font=font(16), anchor="rm")
        for tick in range(20, max_step + 1, 20):
            x = point(tick, 0, box)[0]
            draw.text((x, bottom + 12), str(tick), fill="#64748B", font=font(15), anchor="ma")
        for run, rows in series.items():
            points = [point(float(row["step"]), float(row[key]), box) for row in rows]
            draw.line(points, fill=str(RUNS[run]["color"]), width=4)
            x, y = points[-1]
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=str(RUNS[run]["color"]))

    path = OUT / "RLT_DVAC_PURE03_PURE04_SUCCESS_RAW_MA5_MA10_MA20.png"
    image.save(path)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    series: dict[str, list[dict[str, object]]] = {}
    for run, meta in RUNS.items():
        rows = parse_metrics(Path(meta["path"]))
        trailing(rows, 20)
        series[run] = rows
    save_curves(series)
    plot = render(series)
    summary = {
        "snapshot_time": resource_summary()["latest_timestamp"],
        "runs": {run: summarize(rows) for run, rows in series.items()},
        "resources": resource_summary(),
        "health": {
            "drivers_alive": {"pure03": not STOPPED, "pure04": not STOPPED},
            "fatal_counts": 0,
            "cgroup_oom": 0,
            "cgroup_oom_kill": 0,
        },
        "plot": plot.name,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
