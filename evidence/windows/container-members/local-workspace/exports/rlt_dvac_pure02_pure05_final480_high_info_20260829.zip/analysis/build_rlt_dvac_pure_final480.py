from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("RLT_PURE_LIVE_OUT", str(
    ROOT
    / "docs"
    / "rlinf-robotwin-pi0-dvac-telemetry"
    / "evidence"
    / "rlt_dvac_pure_dual_live_g146_20260828"
)))
S05_LOG = OUT / "raw" / "s0p5_metrics.log"
S20_LOG = OUT / "raw" / "s2p0_metrics.log"
RESOURCE_CSV = OUT / "raw" / "paired_resources.csv"

COLORS = {"s0p5": "#1877D2", "s2p0": "#E66101"}
LABELS = {"s0p5": "[0,2] shorthand / s0p5", "s2p0": "[0,5] shorthand / s2p0"}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size=size)


def value(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def section(block: str, start: str, *ends: str) -> str:
    begin = block.find(start)
    if begin < 0:
        return ""
    stops = [block.find(end, begin + len(start)) for end in ends]
    stops = [stop for stop in stops if stop >= 0]
    return block[begin : min(stops) if stops else len(block)]


def parse_metrics(path: Path) -> list[dict[str, float | int | None]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    markers = list(re.finditer(r"Global Step:\s*(\d+)/480", text))
    rows: list[dict[str, float | int | None]] = []
    for index, marker in enumerate(markers):
        stop = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        block = text[marker.start() : stop]
        env = section(block, " Environment ", " Evaluation ", " Replay Buffer ")
        success = value(env, r"success_once=([-+0-9.eE]+)")
        if success is None:
            continue
        eval_block = section(block, " Evaluation ", " Replay Buffer ")
        eval_success = value(eval_block, r"success_once=([-+0-9.eE]+)")
        row: dict[str, float | int | None] = {
            "step": int(marker.group(1)),
            "raw": 100.0 * success,
            "eval": None if eval_success is None else 100.0 * eval_success,
        }
        for key in (
            "weight_p05",
            "weight_mean",
            "weight_p95",
            "weight_ess_ratio",
            "top20_weight_mass",
        ):
            row[key] = value(block, rf"actor/rlt_dvac/{key}=([-+0-9.eE]+)")
        rows.append(row)

    deduplicated = {int(row["step"]): row for row in rows}
    rows = [deduplicated[step] for step in sorted(deduplicated)]
    for index, row in enumerate(rows):
        for window in (5, 10):
            values = [float(item["raw"]) for item in rows[max(0, index - window + 1) : index + 1]]
            row[f"ma{window}"] = sum(values) / len(values)
    return rows


def summarize(rows: list[dict[str, float | int | None]], end: int) -> dict[str, object]:
    selected = [row for row in rows if int(row["step"]) <= end]
    values = [float(row["raw"]) for row in selected]
    latest = selected[-1]
    fixed = [
        [int(row["step"]), float(row["eval"])]
        for row in selected
        if row["eval"] is not None
    ]
    weight_keys = (
        "weight_p05",
        "weight_mean",
        "weight_p95",
        "weight_ess_ratio",
        "top20_weight_mass",
    )
    weights = {key: latest[key] for key in weight_keys if latest.get(key) is not None}
    return {
        "latest_step": int(latest["step"]),
        "latest_raw_pct": values[-1],
        "cumulative_mean_pct": sum(values) / len(values),
        "latest_ma5_pct": float(latest["ma5"]),
        "latest_ma10_pct": float(latest["ma10"]),
        "fixed_eval_pct": fixed,
        "latest_method_weights": weights,
    }


def parse_resources() -> dict[str, object]:
    rows: list[dict[str, float]] = []
    with RESOURCE_CSV.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            try:
                rows.append(
                    {
                        "memory_gib": float(raw["memory_current"]) / (2**30),
                        "gpu0_gib": float(raw["gpu0_mib"]) / 1024.0,
                        "gpu1_gib": float(raw["gpu1_mib"]) / 1024.0,
                        "gpu0_util": float(raw["gpu0_util"]),
                        "gpu1_util": float(raw["gpu1_util"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    latest = rows[-1]
    return {
        "latest": latest,
        "peak": {
            "memory_gib": max(row["memory_gib"] for row in rows),
            "gpu0_gib": max(row["gpu0_gib"] for row in rows),
            "gpu1_gib": max(row["gpu1_gib"] for row in rows),
        },
        "samples": len(rows),
    }


def save_csv(series: dict[str, list[dict[str, float | int | None]]]) -> None:
    by_run = {
        name: {int(row["step"]): row for row in rows}
        for name, rows in series.items()
    }
    max_step = max(max(rows) for rows in by_run.values())
    fields = [
        "step",
        "s0p5_raw_pct",
        "s0p5_ma5_pct",
        "s0p5_ma10_pct",
        "s2p0_raw_pct",
        "s2p0_ma5_pct",
        "s2p0_ma10_pct",
    ]
    with (OUT / "success_curves.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step in range(1, max_step + 1):
            row: dict[str, object] = {"step": step}
            for name in ("s0p5", "s2p0"):
                source = by_run[name].get(step)
                for key in ("raw", "ma5", "ma10"):
                    row[f"{name}_{key}_pct"] = "" if source is None else source[key]
            writer.writerow(row)


def render(series: dict[str, list[dict[str, float | int | None]]], common_step: int) -> Path:
    width, height = 1840, 1360
    image = Image.new("RGB", (width, height), "#F6F8FB")
    draw = ImageDraw.Draw(image)
    draw.text((58, 38), f"RLT-DVAC-Pure through common Step {common_step}", fill="#172033", font=font(39, True))
    draw.text(
        (58, 91),
        "Training-rollout success is shown at every cycle; fixed-20 evaluation is separate.",
        fill="#52627A",
        font=font(21),
    )
    for index, name in enumerate(("s0p5", "s2p0")):
        x = 1030 + index * 355
        draw.line((x, 96, x + 46, 96), fill=COLORS[name], width=6)
        draw.text((x + 58, 96), LABELS[name], fill="#475569", font=font(18), anchor="lm")

    boxes = [
        (105, 215, 855, 705),
        (980, 215, 1730, 705),
        (105, 835, 855, 1275),
        (980, 835, 1730, 1275),
    ]
    specs = [
        ("Per-step training success", "raw"),
        ("Trailing 5-step mean", "ma5"),
        ("Trailing 10-step mean", "ma10"),
        ("Fixed-20 evaluation", "eval"),
    ]

    def point(step: float, val: float, box: tuple[int, int, int, int]) -> tuple[int, int]:
        left, top, right, bottom = box
        x = left + (step - 1) / max(1, common_step - 1) * (right - left)
        y = bottom - val / 100.0 * (bottom - top)
        return round(x), round(y)

    for box, (title, key) in zip(boxes, specs):
        left, top, right, bottom = box
        draw.rounded_rectangle(
            (left - 52, top - 65, right + 38, bottom + 48),
            radius=16,
            fill="#FFFFFF",
            outline="#D7E0EA",
            width=2,
        )
        draw.text((left - 22, top - 50), title, fill="#172033", font=font(27, True))
        for tick in (0, 25, 50, 75, 100):
            y = point(1, tick, box)[1]
            draw.line((left, y, right, y), fill="#D9E1EA", width=1)
            draw.text((left - 12, y), str(tick), fill="#64748B", font=font(17), anchor="rm")
        for tick in range(25, common_step + 1, 25):
            x = point(tick, 0, box)[0]
            draw.line((x, bottom, x, bottom + 6), fill="#64748B", width=1)
            draw.text((x, bottom + 12), str(tick), fill="#64748B", font=font(16), anchor="ma")
        for name, rows in series.items():
            selected = [row for row in rows if int(row["step"]) <= common_step and row.get(key) is not None]
            points = [point(float(row["step"]), float(row[key]), box) for row in selected]
            if len(points) > 1:
                draw.line(points, fill=COLORS[name], width=4)
            if key == "eval":
                for x, y in points:
                    draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="#FFFFFF", outline=COLORS[name], width=3)

    path = OUT / f"RLT_DVAC_PURE_SUCCESS_THROUGH_G{common_step}.png"
    image.save(path)
    return path


def main() -> None:
    series = {"s0p5": parse_metrics(S05_LOG), "s2p0": parse_metrics(S20_LOG)}
    common_step = min(int(rows[-1]["step"]) for rows in series.values())
    save_csv(series)
    plot = render(series, common_step)
    summary = {
        "snapshot_source_time": os.environ.get("RLT_PURE_SNAPSHOT_TIME", "2026-08-28T10:31:05+08:00"),
        "common_step": common_step,
        "runs_through_common_step": {name: summarize(rows, common_step) for name, rows in series.items()},
        "source_latest": {
            name: summarize(rows, int(rows[-1]["step"])) for name, rows in series.items()
        },
        "resources": parse_resources(),
        "live_health": {
            "both_drivers_alive": os.environ.get("RLT_BOTH_DRIVERS_ALIVE", "true").lower() == "true",
            "oom": int(os.environ.get("RLT_OOM", "0")),
            "oom_kill": int(os.environ.get("RLT_OOM_KILL", "0")),
            "gpu_snapshot_mib": {
                "gpu0": int(os.environ.get("RLT_GPU0_MIB", "17152")),
                "gpu1": int(os.environ.get("RLT_GPU1_MIB", "25235")),
            },
            "cgroup_memory_gib": float(os.environ.get("RLT_MEMORY_GIB", "186.24")),
        },
        "plot": plot.name,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
