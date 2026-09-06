"""Render a compact host/GPU resource view for the current SZ training pair."""

from __future__ import annotations

import csv
import importlib.util
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(r"C:\Users\86136\Documents\rl")
OUT = ROOT / "docs/rlinf-shenzhen-grpo-dvac-action-adv/evidence/current-action-mid-st-narrow-live-20260831-1043"
BASE_SCRIPT = ROOT / "local_scripts/render_shenzhen_grpo_vs_ppo_20260823.py"
TOTAL_GIB = 2113418768 / 1048576
RAY_AVAILABLE_LINE_GIB = TOTAL_GIB * 0.05


def load_base():
    spec = importlib.util.spec_from_file_location("plot_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("plot helper unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_rows(path: Path, gpu_fields: tuple[str, str]) -> list[dict[str, float | datetime]]:
    rows: list[dict[str, float | datetime]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "time": datetime.fromisoformat(row["timestamp"]),
                    "available": float(row["host_mem_available_kib"]) / 1048576,
                    "gpu": (float(row[gpu_fields[0]]) + float(row[gpu_fields[1]])) / 2048,
                }
            )
    return rows


def draw_panel(
    draw: ImageDraw.ImageDraw,
    base,
    box: tuple[int, int, int, int],
    *,
    title: str,
    series: list[tuple[str, str, list[tuple[float, float]]]],
    y_min: float,
    y_max: float,
    x_max: float,
    unit: str,
    threshold: float | None = None,
) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=20, fill=base.WHITE, outline=base.GRID, width=2)
    draw.text((left + 24, top + 18), title, fill=base.INK, font=base.F_PANEL)
    pl, pt, pr, pb = left + 105, top + 76, right - 38, bottom - 62

    def px(value: float) -> float:
        return pl + value / max(x_max, 1e-6) * (pr - pl)

    def py(value: float) -> float:
        return pb - (value - y_min) / (y_max - y_min) * (pb - pt)

    for idx in range(6):
        value = y_min + (y_max - y_min) * idx / 5
        y = py(value)
        draw.line((pl, y, pr, y), fill=base.GRID, width=2)
        label = f"{value:.0f} {unit}"
        bbox = draw.textbbox((0, 0), label, font=base.F_AXIS)
        draw.text((pl - 12 - (bbox[2] - bbox[0]), y - 11), label, fill=base.GRAY, font=base.F_AXIS)
    for idx in range(6):
        value = x_max * idx / 5
        x = px(value)
        draw.line((x, pb, x, pb + 8), fill=base.GRAY, width=2)
        label = f"{value:.0f}h"
        bbox = draw.textbbox((0, 0), label, font=base.F_AXIS)
        draw.text((x - (bbox[2] - bbox[0]) / 2, pb + 12), label, fill=base.GRAY, font=base.F_AXIS)
    if threshold is not None:
        y = py(threshold)
        draw.line((pl, y, pr, y), fill="#DC2626", width=3)
        draw.text((pl + 12, y - 31), "approx. Ray 95% line", fill="#DC2626", font=base.F_SMALL)
    for name, color, points in series:
        scaled = [(px(x), py(y)) for x, y in points]
        draw.line(scaled, fill=color, width=4, joint="curve")
        x, y = scaled[-1]
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline="white", width=2)
    lx = left + 28
    for name, color, _ in series:
        draw.line((lx, top + 62, lx + 42, top + 62), fill=color, width=5)
        draw.text((lx + 52, top + 48), name, fill=color, font=base.F_LEGEND)
        lx += 470
    draw.line((pl, pt, pl, pb), fill=base.GRAY, width=2)
    draw.line((pl, pb, pr, pb), fill=base.GRAY, width=2)


def main() -> None:
    base = load_base()
    action = load_rows(OUT / "raw/action_mid/runtime/resource.csv", ("gpu4_used_mib", "gpu5_used_mib"))
    st = load_rows(OUT / "raw/st_narrow/runtime/resource.csv", ("gpu6_used_mib", "gpu7_used_mib"))
    start = min(action[0]["time"], st[0]["time"])

    def points(rows: list[dict[str, float | datetime]], key: str) -> list[tuple[float, float]]:
        return [
            ((row["time"] - start).total_seconds() / 3600, float(row[key]))
            for row in rows
        ]

    x_max = max(points(action, "available")[-1][0], points(st, "available")[-1][0])
    image = Image.new("RGB", (1600, 1080), base.LIGHT)
    draw = base.add_header(
        image,
        "SZ-H100 current pair — resource trend",
        "Read-only capture through 2026-08-31 10:45 CST; GPU values are pair averages.",
    )
    draw_panel(
        draw,
        base,
        (45, 145, 1555, 575),
        title="Host available memory",
        series=[("host available", "#111827", points(action, "available"))],
        y_min=0,
        y_max=2050,
        x_max=x_max,
        unit="GiB",
        threshold=RAY_AVAILABLE_LINE_GIB,
    )
    draw_panel(
        draw,
        base,
        (45, 605, 1555, 1035),
        title="Average GPU memory per card",
        series=[
            ("Action-Adv GPU4/5", "#E69F00", points(action, "gpu")),
            ("ST-DVAC GPU6/7", "#00796B", points(st, "gpu")),
        ],
        y_min=0,
        y_max=80,
        x_max=x_max,
        unit="GiB",
    )
    path = OUT / "02_current_pair_resource_trend.png"
    image.save(path, optimize=True)
    print(path)


if __name__ == "__main__":
    main()
