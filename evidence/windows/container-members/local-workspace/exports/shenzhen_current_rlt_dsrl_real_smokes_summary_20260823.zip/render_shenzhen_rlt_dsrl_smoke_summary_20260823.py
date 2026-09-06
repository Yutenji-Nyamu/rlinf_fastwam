#!/usr/bin/env python3
"""Render a mobile-readable RLT/DSRL smoke summary from observer CSV evidence.

Only Pillow is used so the renderer runs in the bundled desktop Python. GPU
memory curves contain exactly the rows observed in the five downloaded CSVs;
there is no interpolation or synthetic sampling.
"""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "rlinf-shenzhen-rlt-dsrl-port" / "evidence" / "real-smokes-20260823"
EXTRACTED = EVIDENCE / "extracted"
OUTPUT = EVIDENCE / "01_rlt_dsrl_smoke_resource_and_time.png"

CSV_SPECS = [
    (
        "RLT Stage 1｜fresh｜2 steps",
        EXTRACTED / "rlinf-rlt" / "smoke-stage1-current-ar-2step-20260823" / "runtime" / "resource.csv",
        (4, 5),
        "离线 AR reconstruction；第 2 步含 checkpoint 保存",
    ),
    (
        "RLT Stage 2｜fresh｜cycle 1",
        EXTRACTED / "rlinf-rlt" / "smoke-stage2-current-ar-fresh1-resume1-20260823" / "fresh" / "resource.csv",
        (4, 5),
        "4 train env + 4 fixed-eval env",
    ),
    (
        "RLT Stage 2｜fresh-process resume｜cycle 2",
        EXTRACTED / "rlinf-rlt" / "smoke-stage2-current-ar-fresh1-resume1-20260823" / "resume" / "resource.csv",
        (4, 5),
        "从 global_step_1 恢复；算法 sidecar 同步恢复",
    ),
    (
        "DSRL｜fresh｜step 1",
        EXTRACTED / "rlinf-current-dsrl" / "smoke-dsrl-pi0-robotwin-2gpu4env-warm4-20260823" / "fresh" / "resource.csv",
        (6, 7),
        "4 env；40 macro transitions；800 optimizer updates",
    ),
    (
        "DSRL｜fresh-process resume｜step 2",
        EXTRACTED / "rlinf-current-dsrl" / "smoke-dsrl-pi0-robotwin-2gpu4env-warm4-20260823" / "resume" / "resource.csv",
        (6, 7),
        "新增 36 macro transitions；720 optimizer updates",
    ),
]

COLORS = {
    "ink": "#152331",
    "muted": "#526170",
    "subtle": "#77838F",
    "grid": "#D9E1E8",
    "axis": "#9AA8B5",
    "card": "#FAFCFE",
    "card_edge": "#D9E2EA",
    "gpu_a": "#0072B2",
    "gpu_b": "#D55E00",
    "rollout": "#56B4E9",
    "eval": "#009E73",
    "train": "#E69F00",
    "other": "#C7CDD4",
    "total": "#665191",
    "warning": "#8A4B08",
}

FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/msyhbd.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
]
FONT_PATH = next((path for path in FONT_CANDIDATES if path.exists()), None)
if FONT_PATH is None:
    raise FileNotFoundError("No Chinese-capable Windows font found")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    if bold:
        bold_path = Path("C:/Windows/Fonts/msyhbd.ttc")
        if bold_path.exists():
            return ImageFont.truetype(str(bold_path), size)
    return ImageFont.truetype(str(FONT_PATH), size)


F_TITLE = font(46, bold=True)
F_SUBTITLE = font(24)
F_CARD_TITLE = font(31, bold=True)
F_SMALL = font(19)
F_TINY = font(17)
F_BAR_TITLE = font(27, bold=True)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def read_resource_csv(path: Path, gpu_ids: tuple[int, int]):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    stamps = [datetime.fromisoformat(row["timestamp"]) for row in rows]
    t0 = stamps[0]
    elapsed = [(stamp - t0).total_seconds() for stamp in stamps]
    series = {}
    for gpu_id in gpu_ids:
        key = f"gpu{gpu_id}_used_mib"
        # The terminal observer row is intentionally retained in the CSV even
        # when nvidia-smi has already disappeared. Omit that blank GPU cell;
        # never turn it into a synthetic zero.
        series[gpu_id] = [
            (t, float(row[key]) / 1024.0)
            for t, row in zip(elapsed, rows)
            if row.get(key, "").strip()
        ]
    return elapsed, series


def nice_ymax(value: float) -> float:
    return max(10.0, math.ceil((value * 1.22) / 5.0) * 5.0)


def draw_resource_card(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    top: int,
    title: str,
    csv_path: Path,
    gpu_ids: tuple[int, int],
    note: str,
) -> None:
    left, right, height = 70, image.width - 70, 455
    bottom = top + height
    draw.rounded_rectangle((left, top, right, bottom), radius=24, fill=COLORS["card"], outline=COLORS["card_edge"], width=2)
    draw.text((left + 30, top + 22), title, font=F_CARD_TITLE, fill=COLORS["ink"])

    elapsed, series = read_resource_csv(csv_path, gpu_ids)
    max_value = max(max(value for _, value in points) for points in series.values())
    ymax = nice_ymax(max_value)
    valid_points = min(len(points) for points in series.values())
    note_text = f"{note}｜GPU 有效采样 {valid_points} 点"
    note_w, _ = text_size(draw, note_text, F_SMALL)
    draw.text((right - 30 - note_w, top + 31), note_text, font=F_SMALL, fill=COLORS["muted"])

    plot_left, plot_right = left + 125, right - 35
    plot_top, plot_bottom = top + 105, bottom - 70
    plot_w, plot_h = plot_right - plot_left, plot_bottom - plot_top
    max_t = max(t for points in series.values() for t, _ in points) or 1.0

    # Axis guides only; each data polyline still uses every observed row once.
    for i in range(5):
        y_value = ymax * i / 4
        py = round(plot_bottom - (y_value / ymax) * plot_h)
        draw.line((plot_left, py, plot_right, py), fill=COLORS["grid"], width=2)
        label = f"{y_value:.0f}"
        label_w, label_h = text_size(draw, label, F_TINY)
        draw.text((plot_left - label_w - 16, py - label_h // 2), label, font=F_TINY, fill=COLORS["muted"])
    for i in range(5):
        t_value = max_t * i / 4
        px = round(plot_left + (t_value / max_t) * plot_w)
        draw.line((px, plot_top, px, plot_bottom), fill=COLORS["grid"], width=2)
        label = f"{t_value:.0f}s"
        label_w, _ = text_size(draw, label, F_TINY)
        draw.text((px - label_w // 2, plot_bottom + 13), label, font=F_TINY, fill=COLORS["muted"])
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=COLORS["axis"], width=3)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=COLORS["axis"], width=3)

    legend_x = plot_left
    for idx, gpu_id in enumerate(gpu_ids):
        samples = series[gpu_id]
        values = [value for _, value in samples]
        color = COLORS["gpu_a"] if idx == 0 else COLORS["gpu_b"]
        points = [
            (
                round(plot_left + (t / max_t) * plot_w),
                round(plot_bottom - (value / ymax) * plot_h),
            )
            for t, value in samples
        ]
        if len(points) > 1:
            draw.line(points, fill=color, width=5, joint="curve")
        for px, py in points:
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color)

        peak_idx = max(range(len(values)), key=values.__getitem__)
        peak_x, peak_y = points[peak_idx]
        draw.ellipse((peak_x - 8, peak_y - 8, peak_x + 8, peak_y + 8), fill=color, outline="white", width=3)
        peak_label = f"{values[peak_idx]:.1f} GiB"
        label_w, label_h = text_size(draw, peak_label, F_SMALL)
        label_x = min(max(peak_x + 12, plot_left), plot_right - label_w - 12)
        label_y = max(plot_top + 4, peak_y - 31 - idx * 29)
        draw.rounded_rectangle((label_x - 7, label_y - 4, label_x + label_w + 7, label_y + label_h + 5), radius=8, fill="white", outline=color, width=2)
        draw.text((label_x, label_y), peak_label, font=F_SMALL, fill=color)

        draw.line((legend_x, bottom - 30, legend_x + 36, bottom - 30), fill=color, width=6)
        legend_text = f"物理 GPU {gpu_id}"
        draw.text((legend_x + 47, bottom - 43), legend_text, font=F_SMALL, fill=COLORS["ink"])
        legend_w, _ = text_size(draw, legend_text, F_SMALL)
        legend_x += legend_w + 105

    draw.text((left + 22, plot_top + plot_h // 2 - 15), "GiB", font=F_TINY, fill=COLORS["muted"])


def draw_legend(draw: ImageDraw.ImageDraw, x: int, y: int, items: list[tuple[str, str]]) -> None:
    cursor = x
    for label, color in items:
        draw.rounded_rectangle((cursor, y + 3, cursor + 26, y + 23), radius=4, fill=color)
        draw.text((cursor + 36, y), label, font=F_TINY, fill=COLORS["muted"])
        width, _ = text_size(draw, label, F_TINY)
        cursor += width + 75


def draw_stacked_row(
    draw: ImageDraw.ImageDraw,
    label: str,
    values: list[tuple[float, str]],
    total: float,
    x: int,
    y: int,
    width: int,
    scale_max: float,
) -> None:
    draw.text((x, y - 1), label, font=F_SMALL, fill=COLORS["ink"])
    bar_x, bar_y, bar_h = x + 250, y, 42
    cursor = bar_x
    for value, color in values:
        segment_w = round((value / scale_max) * width)
        if segment_w > 0:
            draw.rectangle((cursor, bar_y, cursor + segment_w, bar_y + bar_h), fill=color)
        cursor += segment_w
    draw.rounded_rectangle((bar_x, bar_y, bar_x + round((total / scale_max) * width), bar_y + bar_h), radius=5, outline="#8895A2", width=2)
    draw.text((bar_x + round((total / scale_max) * width) + 16, y + 4), f"总计 {total:.1f}s", font=F_SMALL, fill=COLORS["ink"])


def draw_time_block(image: Image.Image, draw: ImageDraw.ImageDraw, top: int) -> None:
    left, right, bottom = 70, image.width - 70, image.height - 70
    draw.rounded_rectangle((left, top, right, bottom), radius=24, fill="#F8FAFC", outline=COLORS["card_edge"], width=2)
    draw.text((left + 30, top + 25), "Step / cycle 时间分解（日志实测）", font=F_CARD_TITLE, fill=COLORS["ink"])
    draw.text(
        (left + 30, top + 72),
        "“其余” = total − 已记录阶段，仅表示未细分 wall time；不把它擅自归因为某一个组件。",
        font=F_SMALL,
        fill=COLORS["muted"],
    )

    y = top + 125
    draw.text((left + 30, y), "RLT Stage 1｜两步总时长", font=F_BAR_TITLE, fill=COLORS["ink"])
    bar_x, bar_w = left + 340, 580
    for row_label, value, color, dy in [
        ("Step 1", 20.7, "#7AA6C2", 0),
        ("Step 2（含保存）", 23.7, COLORS["total"], 58),
    ]:
        row_y = y + 48 + dy
        draw.text((left + 30, row_y + 6), row_label, font=F_SMALL, fill=COLORS["ink"])
        width = round((value / 28.0) * bar_w)
        draw.rounded_rectangle((bar_x, row_y, bar_x + width, row_y + 42), radius=6, fill=color)
        draw.text((bar_x + width + 14, row_y + 5), f"{value:.1f}s", font=F_SMALL, fill=COLORS["ink"])
    draw.text((left + 1010, y + 61), "2-step smoke 保存频率高，", font=F_SMALL, fill=COLORS["warning"])
    draw.text((left + 1010, y + 94), "不能据此外推 2k-step formal 吞吐。", font=F_SMALL, fill=COLORS["warning"])

    y2 = y + 190
    draw.text((left + 30, y2), "RLT Stage 2｜单 cycle（横轴上限 70 秒）", font=F_BAR_TITLE, fill=COLORS["ink"])
    draw_legend(draw, left + 720, y2 + 4, [("rollout", COLORS["rollout"]), ("fixed eval", COLORS["eval"]), ("训练", COLORS["train"]), ("其余", COLORS["other"])])
    for label, rollout, evaluation, train, total, dy in [
        ("fresh", 20.371, 18.280, 0.743, 60.067, 48),
        ("resume", 23.191, 36.197, 1.121, 62.964, 108),
    ]:
        other = max(0.0, total - rollout - evaluation - train)
        draw_stacked_row(
            draw,
            label,
            [(rollout, COLORS["rollout"]), (evaluation, COLORS["eval"]), (train, COLORS["train"]), (other, COLORS["other"])],
            total,
            left + 30,
            y2 + dy,
            1050,
            70.0,
        )

    y3 = y2 + 190
    draw.text((left + 30, y3), "DSRL｜单 step（横轴上限 450 秒）", font=F_BAR_TITLE, fill=COLORS["ink"])
    draw_legend(draw, left + 795, y3 + 4, [("rollout", COLORS["rollout"]), ("UTD20 训练", COLORS["train"]), ("其余", COLORS["other"])])
    for label, rollout, train, total, dy in [
        ("fresh", 31.533, 338.100, 429.239, 48),
        ("resume", 34.598, 307.900, 375.282, 108),
    ]:
        other = max(0.0, total - rollout - train)
        draw_stacked_row(
            draw,
            label,
            [(rollout, COLORS["rollout"]), (train, COLORS["train"]), (other, COLORS["other"])],
            total,
            left + 30,
            y3 + dy,
            1050,
            450.0,
        )


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1800, 3600), "white")
    draw = ImageDraw.Draw(image)

    draw.text((70, 52), "深圳 H100｜current RLinf：RLT / DSRL smoke", font=F_TITLE, fill=COLORS["ink"])
    draw.text((70, 118), "资源曲线与单步时间｜fresh + fresh-process resume", font=font(32, bold=True), fill="#2D536E")
    draw.text(
        (70, 174),
        "五组曲线均来自 10 秒级 observer CSV 原始采样：不插值、不补点；显存单位 GiB（CSV MiB ÷ 1024）。",
        font=F_SUBTITLE,
        fill=COLORS["muted"],
    )

    top = 235
    for spec in CSV_SPECS:
        draw_resource_card(image, draw, top, *spec)
        top += 485

    draw_time_block(image, draw, top + 5)
    draw.text((image.width - 650, image.height - 45), "证据快照：2026-08-23｜每次 resume 均为独立进程", font=F_TINY, fill=COLORS["subtle"])
    image.save(OUTPUT, format="PNG", optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
