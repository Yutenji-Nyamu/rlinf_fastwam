from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


WHITE = "#ffffff"
BLACK = "#20242a"
GRID = "#d9dde3"
MUTED = "#68717d"
COLORS = {
    "success": "#23866b",
    "failure": "#c34f4f",
    "raw": "#6741a5",
    "position": "#757b82",
    "residual": "#d77d1f",
    "R": "#2f69ad",
    "S": "#16805f",
    "I": "#d35d12",
}

GROUPS = [
    ("pi0", "adjust_bottle", "pi0 / adjust bottle"),
    ("fastwam", "adjust_bottle", "Fast-WAM / adjust bottle"),
    ("fastwam", "move_stapler_pad", "Fast-WAM / move stapler"),
    ("fastwam", "turn_switch", "Fast-WAM / turn switch"),
    ("fastwam", "pick_diverse_bottles", "Fast-WAM / pick bottles"),
]

CASE_STRIPS = {
    ("pi0", "adjust_bottle", "ep000033_reset100100005"): "pi0__adjust_bottle__run-pi0-adjust_bottle-fi__ep000033_reset100100005.png",
    ("pi0", "adjust_bottle", "ep000050_reset100100017"): "pi0__adjust_bottle__run-pi0-adjust_bottle-fi__ep000050_reset100100017.png",
    ("fastwam", "adjust_bottle", "episode0002_reset1"): "fastwam__adjust_bottle__run-fastwam-adjust_b__episode0002_reset1.png",
    ("fastwam", "move_stapler_pad", "episode0004_reset3"): "fastwam__move_stapler_pad__run-fastwam-move___episode0004_reset3.png",
    ("fastwam", "move_stapler_pad", "episode0006_reset5"): "fastwam__move_stapler_pad__run-fastwam-move___episode0006_reset5.png",
    ("fastwam", "turn_switch", "episode0009_reset8"): "fastwam__turn_switch__run-fastwam-turn_switc__episode0009_reset8.png",
    ("fastwam", "turn_switch", "episode0001_reset0"): "fastwam__turn_switch__run-fastwam-turn_switc__episode0001_reset0.png",
    ("fastwam", "pick_diverse_bottles", "episode0004_reset3"): "fastwam__pick_diverse_bottles__run-fastwam-p__episode0004_reset3.png",
    ("fastwam", "pick_diverse_bottles", "episode0005_reset4"): "fastwam__pick_diverse_bottles__run-fastwam-p__episode0005_reset4.png",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(path), size=size)


FONTS = {
    "title": font(34, True),
    "subtitle": font(25, True),
    "body": font(20),
    "small": font(17),
    "tiny": font(14),
}


def group_label(policy: str, task: str) -> str:
    for candidate_policy, candidate_task, label in GROUPS:
        if policy == candidate_policy and task == candidate_task:
            return label
    return f"{policy} / {task}"


def add_group_label(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["display_group"] = [group_label(str(p), str(t)) for p, t in zip(result["policy"], result["task"])]
    return result


def text_size(draw: ImageDraw.ImageDraw, text: str, which: str = "body") -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=FONTS[which])
    return box[2] - box[0], box[3] - box[1]


def centered_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, which: str, fill: str = BLACK) -> None:
    x0, y0, x1, y1 = box
    width, height = text_size(draw, text, which)
    draw.text(((x0 + x1 - width) / 2, (y0 + y1 - height) / 2), text, font=FONTS[which], fill=fill)


def title(draw: ImageDraw.ImageDraw, width: int, text: str, subtitle: str | None = None) -> int:
    draw.text((42, 26), text, font=FONTS["title"], fill=BLACK)
    y = 74
    if subtitle:
        draw.text((44, y), subtitle, font=FONTS["small"], fill=MUTED)
        y += 35
    draw.line((40, y, width - 40, y), fill=GRID, width=2)
    return y + 24


def panel(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], heading: str | None = None) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = rect
    draw.rounded_rectangle(rect, radius=14, fill="#fbfcfd", outline="#cfd5dc", width=2)
    if heading:
        draw.text((x0 + 18, y0 + 13), heading, font=FONTS["subtitle"], fill=BLACK)
        return (x0 + 55, y0 + 70, x1 - 28, y1 - 48)
    return (x0 + 55, y0 + 32, x1 - 28, y1 - 48)


def map_point(value: float, low: float, high: float, pixel_low: float, pixel_high: float) -> float:
    if high <= low:
        return (pixel_low + pixel_high) / 2
    return pixel_low + (value - low) * (pixel_high - pixel_low) / (high - low)


def draw_axes(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    *,
    x_label: str,
    y_label: str,
    zero: bool = False,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = rect
    plot = (x0 + 58, y0 + 10, x1 - 12, y1 - 42)
    px0, py0, px1, py1 = plot
    for fraction in np.linspace(0, 1, 5):
        yy = int(py1 - fraction * (py1 - py0))
        value = y_min + fraction * (y_max - y_min)
        draw.line((px0, yy, px1, yy), fill=GRID, width=1)
        draw.text((x0, yy - 8), f"{value:.2f}", font=FONTS["tiny"], fill=MUTED)
    draw.line((px0, py0, px0, py1), fill=BLACK, width=2)
    draw.line((px0, py1, px1, py1), fill=BLACK, width=2)
    if zero and y_min < 0 < y_max:
        yy = int(map_point(0, y_min, y_max, py1, py0))
        draw.line((px0, yy, px1, yy), fill="#7d838a", width=2)
    for fraction in np.linspace(0, 1, 5):
        xx = int(px0 + fraction * (px1 - px0))
        value = x_min + fraction * (x_max - x_min)
        draw.text((xx - 16, py1 + 7), f"{value:.0f}", font=FONTS["tiny"], fill=MUTED)
    centered_text(draw, (px0, py1 + 22, px1, y1), x_label, "tiny", MUTED)
    draw.text((x0, y0 - 3), y_label, font=FONTS["tiny"], fill=MUTED)
    return plot


def draw_line_chart(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    series: Iterable[tuple[np.ndarray, np.ndarray, str, str]],
    *,
    x_label: str,
    y_label: str,
    zero: bool = False,
    verticals: Iterable[float] = (),
    forced_y: tuple[float, float] | None = None,
) -> None:
    series = list(series)
    all_x = np.concatenate([np.asarray(x, dtype=float) for x, _, _, _ in series])
    all_y = np.concatenate([np.asarray(y, dtype=float) for _, y, _, _ in series])
    x_min, x_max = float(np.nanmin(all_x)), float(np.nanmax(all_x))
    if forced_y is None:
        y_min, y_max = np.nanquantile(all_y, [0.015, 0.985])
        padding = max(float((y_max - y_min) * 0.12), 0.08)
        y_min, y_max = float(y_min - padding), float(y_max + padding)
        if zero:
            y_min, y_max = min(y_min, -0.05), max(y_max, 0.05)
    else:
        y_min, y_max = forced_y
    plot = draw_axes(draw, rect, x_min, x_max, y_min, y_max, x_label=x_label, y_label=y_label, zero=zero)
    px0, py0, px1, py1 = plot
    for value in verticals:
        xx = int(map_point(float(value), x_min, x_max, px0, px1))
        draw.line((xx, py0, xx, py1), fill="#b4bac1", width=1)
    legend_x = px0 + 8
    for x_values, y_values, color, name in series:
        points = [
            (
                int(map_point(float(x), x_min, x_max, px0, px1)),
                int(map_point(float(y), y_min, y_max, py1, py0)),
            )
            for x, y in zip(x_values, y_values)
            if np.isfinite(x) and np.isfinite(y)
        ]
        if len(points) > 1:
            draw.line(points, fill=color, width=3, joint="curve")
        for point in points[:: max(1, len(points) // 18)]:
            draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=color)
        draw.line((legend_x, py0 + 8, legend_x + 26, py0 + 8), fill=color, width=4)
        draw.text((legend_x + 32, py0 - 1), name, font=FONTS["tiny"], fill=BLACK)
        legend_x += 42 + text_size(draw, name, "tiny")[0]


def draw_histogram(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    groups: list[tuple[np.ndarray, str, str]],
    *,
    x_label: str,
) -> None:
    values = np.concatenate([data for data, _, _ in groups if len(data)])
    bins = np.linspace(float(values.min()), float(values.max()), 19)
    histograms = [(np.histogram(data, bins=bins, density=True)[0], color, label) for data, color, label in groups if len(data)]
    max_density = max(float(hist.max()) for hist, _, _ in histograms) * 1.12
    plot = draw_axes(draw, rect, bins[0], bins[-1], 0, max_density, x_label=x_label, y_label="density")
    px0, py0, px1, py1 = plot
    bin_width = (px1 - px0) / (len(bins) - 1)
    for histogram, color, _ in histograms:
        points: list[tuple[int, int]] = [(px0, py1)]
        for index, value in enumerate(histogram):
            left = int(px0 + index * bin_width)
            right = int(px0 + (index + 1) * bin_width)
            yy = int(map_point(float(value), 0, max_density, py1, py0))
            points.extend([(left, yy), (right, yy)])
        points.append((px1, py1))
        draw.line(points, fill=color, width=4, joint="curve")
    legend_y = py0 + 4
    for data, color, label in groups:
        if not len(data):
            continue
        text = f"{label} n={len(data)}, mean={np.mean(data):.2f}"
        draw.rectangle((px0 + 8, legend_y + 5, px0 + 26, legend_y + 18), fill=color)
        draw.text((px0 + 34, legend_y), text, font=FONTS["tiny"], fill=BLACK)
        legend_y += 24


def diverging_rgb(value: float, limit: float) -> tuple[int, int, int]:
    t = max(-1.0, min(1.0, value / max(limit, 1e-12)))
    white = np.array([247, 247, 247], dtype=float)
    target = np.array([194, 63, 63] if t >= 0 else [52, 99, 173], dtype=float)
    rgb = white * (1 - abs(t)) + target * abs(t)
    return tuple(int(v) for v in rgb)


def sequential_rgb(value: float, low: float, high: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, (value - low) / max(high - low, 1e-12)))
    anchors = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], dtype=float)
    position = t * (len(anchors) - 1)
    index = min(int(position), len(anchors) - 2)
    fraction = position - index
    rgb = anchors[index] * (1 - fraction) + anchors[index + 1] * fraction
    return tuple(int(v) for v in rgb)


def draw_heatmap(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    matrix: np.ndarray,
    *,
    heading: str,
    symmetric: bool,
    gray_last_row: bool = False,
) -> None:
    x0, y0, x1, y1 = rect
    draw.text((x0, y0), heading, font=FONTS["small"], fill=BLACK)
    heat_rect = (x0 + 74, y0 + 34, x1 - 50, y1 - 40)
    hx0, hy0, hx1, hy1 = heat_rect
    finite = matrix[np.isfinite(matrix)]
    if symmetric:
        limit = float(np.quantile(np.abs(finite), 0.98))
        array = np.array([[diverging_rgb(float(value), limit) for value in row] for row in matrix], dtype=np.uint8)
        caption = f"blue<0, red>0; 98% range +/-{limit:.2f}"
    else:
        low, high = np.quantile(finite, [0.02, 0.98])
        array = np.array([[sequential_rgb(float(value), float(low), float(high)) for value in row] for row in matrix], dtype=np.uint8)
        caption = f"2%-98% range {low:.2f}..{high:.2f}"
    heat = Image.fromarray(array, mode="RGB").resize((hx1 - hx0, hy1 - hy0), Image.Resampling.NEAREST)
    if gray_last_row and matrix.shape[0] > 1:
        overlay = Image.new("RGBA", heat.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        row_height = heat.height / matrix.shape[0]
        overlay_draw.rectangle((0, int((matrix.shape[0] - 1) * row_height), heat.width, heat.height), fill=(140, 140, 140, 95))
        heat = Image.alpha_composite(heat.convert("RGBA"), overlay).convert("RGB")
    image.paste(heat, (hx0, hy0))
    draw.rectangle(heat_rect, outline=BLACK, width=2)
    draw.text((hx0, hy1 + 8), "future position h", font=FONTS["tiny"], fill=MUTED)
    draw.text((x0, hy0), "q", font=FONTS["tiny"], fill=MUTED)
    draw.text((x1 - 315, y0 + 5), caption, font=FONTS["tiny"], fill=MUTED)


def paste_strip(image: Image.Image, rect: tuple[int, int, int, int], path: Path) -> None:
    source = Image.open(path).convert("RGB")
    x0, y0, x1, y1 = rect
    scale = min((x1 - x0) / source.width, (y1 - y0) / source.height)
    resized = source.resize((max(1, int(source.width * scale)), max(1, int(source.height * scale))), Image.Resampling.LANCZOS)
    image.paste(resized, (x0 + (x1 - x0 - resized.width) // 2, y0 + (y1 - y0 - resized.height) // 2))


def bootstrap_difference(success: np.ndarray, failure: np.ndarray, seed: int, draws: int = 10_000) -> tuple[float, float, float]:
    success = np.asarray(success, dtype=float)
    failure = np.asarray(failure, dtype=float)
    center = float(success.mean() - failure.mean())
    rng = np.random.default_rng(seed)
    sampled_success = rng.choice(success, size=(draws, len(success)), replace=True).mean(axis=1)
    sampled_failure = rng.choice(failure, size=(draws, len(failure)), replace=True).mean(axis=1)
    low, high = np.quantile(sampled_success - sampled_failure, [0.025, 0.975])
    return center, float(low), float(high)


def summarize_differences(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for group_index, (_, _, label) in enumerate(GROUPS):
        group = frame[frame["display_group"].eq(label)]
        success = group[group["success"].eq(True)]
        failure = group[group["success"].eq(False)]
        if success.empty or failure.empty:
            continue
        for metric_index, metric in enumerate(metrics):
            center, low, high = bootstrap_difference(
                success[metric].to_numpy(float),
                failure[metric].to_numpy(float),
                seed=20260823 + group_index * 100 + metric_index,
            )
            rows.append(
                {
                    "display_group": label,
                    "metric": metric,
                    "success_episodes": len(success),
                    "failure_episodes": len(failure),
                    "success_mean": float(success[metric].mean()),
                    "failure_mean": float(failure[metric].mean()),
                    "success_minus_failure": center,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def draw_interval_panel(
    image: Image.Image,
    rect: tuple[int, int, int, int],
    summary: pd.DataFrame,
    metric: str,
    heading: str,
    color: str,
) -> None:
    draw = ImageDraw.Draw(image)
    inner = panel(draw, rect, heading)
    frame = summary[summary["metric"].eq(metric)].copy()
    order = [label for _, _, label in GROUPS if label != "Fast-WAM / adjust bottle"]
    frame = frame.set_index("display_group").reindex(order)
    values = frame[["bootstrap_ci_low", "bootstrap_ci_high"]].to_numpy(float)
    low = min(float(np.nanmin(values)), -0.05)
    high = max(float(np.nanmax(values)), 0.05)
    padding = 0.12 * (high - low)
    low, high = low - padding, high + padding
    x0, y0, x1, y1 = inner
    label_width = 245
    px0, px1 = x0 + label_width, x1 - 20
    zero_x = int(map_point(0, low, high, px0, px1))
    draw.line((zero_x, y0, zero_x, y1), fill="#7b8188", width=2)
    for index, label in enumerate(order):
        yy = int(y0 + (index + 0.5) * (y1 - y0) / len(order))
        row = frame.loc[label]
        draw.text((x0, yy - 11), label.replace("Fast-WAM", "FW"), font=FONTS["small"], fill=BLACK)
        left = int(map_point(float(row["bootstrap_ci_low"]), low, high, px0, px1))
        right = int(map_point(float(row["bootstrap_ci_high"]), low, high, px0, px1))
        center = int(map_point(float(row["success_minus_failure"]), low, high, px0, px1))
        draw.line((left, yy, right, yy), fill=color, width=4)
        draw.line((left, yy - 7, left, yy + 7), fill=color, width=3)
        draw.line((right, yy - 7, right, yy + 7), fill=color, width=3)
        draw.ellipse((center - 7, yy - 7, center + 7, yy + 7), fill=color)
        draw.text((center - 30, yy - 32), f"{float(row['success_minus_failure']):+.3f}", font=FONTS["tiny"], fill=color)
    centered_text(draw, (px0, y1 + 8, px1, rect[3] - 5), "success - failure (95% episode bootstrap CI)", "tiny", MUTED)


def render_signal_map(output: Path) -> Path:
    image = Image.new("RGB", (2300, 1050), WHITE)
    draw = ImageDraw.Draw(image)
    y = title(draw, image.width, "DVAC signal family: six behavior views, not seven independent metrics")

    def box(rect: tuple[int, int, int, int], heading: str, formula: str, color: str) -> None:
        draw.rounded_rectangle(rect, radius=18, fill="#f7f8fa", outline=color, width=4)
        centered_text(draw, (rect[0], rect[1] + 12, rect[2], rect[1] + 58), heading, "subtitle")
        centered_text(draw, (rect[0], rect[1] + 70, rect[2], rect[3] - 8), formula, "body", color)

    def arrow(start: tuple[int, int], end: tuple[int, int], label: str = "") -> None:
        draw.line((*start, *end), fill="#555b62", width=4)
        draw.polygon([(end[0], end[1]), (end[0] - 13, end[1] - 8), (end[0] - 13, end[1] + 8)], fill="#555b62")
        if label:
            draw.text(((start[0] + end[0]) // 2 - 60, (start[1] + end[1]) // 2 - 28), label, font=FONTS["tiny"], fill=MUTED)

    box((60, y + 105, 360, y + 265), "Raw endpoint variance", "V_L(q,h) > 0", COLORS["raw"])
    box((470, y + 105, 770, y + 265), "log-DVAC", "y = ln(V_L + eps)", COLORS["raw"])
    box((920, y + 25, 1290, y + 175), "Position channel", "b_h = mu + P_h", COLORS["position"])
    box((920, y + 260, 1290, y + 410), "Raw residual", "r = y - b_h", COLORS["residual"])
    box((1450, y + 260, 1840, y + 410), "Standardized residual", "R = r / s_h", COLORS["R"])
    box((920, y + 545, 1290, y + 695), "Query-wide shift", "S_raw = mean_h(r)", COLORS["S"])
    box((1450, y + 545, 1840, y + 695), "Local interaction", "I_raw = r - S_raw", COLORS["I"])
    box((1450, y + 45, 1840, y + 175), "Reference scale", "s_h = 1.4826 MAD_h", "#8a8f95")
    arrow((360, y + 185), (470, y + 185), "log")
    arrow((770, y + 175), (920, y + 100), "median over q")
    arrow((770, y + 215), (920, y + 335), "subtract b_h")
    arrow((1290, y + 335), (1450, y + 335), "divide by s_h")
    arrow((1105, y + 410), (1105, y + 545), "mean over h")
    arrow((1290, y + 365), (1450, y + 620), "remove query mean")
    arrow((1645, y + 175), (1645, y + 260))
    draw.rounded_rectangle((1880, y + 80, 2240, y + 670), radius=16, fill="#f2f5f8", outline="#c7ced6", width=2)
    notes = [
        "Six views used in this report:",
        "1  y: raw log-DVAC",
        "2  b/P: position",
        "3  r: raw residual",
        "4  R: standardized residual",
        "5  S: query-wide shift",
        "6  I: local interaction",
        "",
        "mu and s_h are references.",
        "R = S_std + I_std exactly.",
    ]
    for index, line in enumerate(notes):
        draw.text((1905, y + 110 + index * 46), line, font=FONTS["body" if index else "subtitle"], fill=BLACK if index else COLORS["R"])
    path = output / "00_signal_family_map.png"
    image.save(path)
    return path


def render_raw(query: pd.DataFrame, sensitivity: pd.DataFrame, output: Path) -> Path:
    image = Image.new("RGB", (2500, 1150), WHITE)
    draw = ImageDraw.Draw(image)
    y = title(draw, image.width, "Signal 1 — raw DVAC / log-DVAC", "Each panel owns its x-axis: compare patterns within a run, not absolute pi0-versus-Fast-WAM levels.")
    query = add_group_label(query)
    sensitivity = add_group_label(sensitivity)
    margin, gap = 28, 18
    width = (image.width - 2 * margin - 4 * gap) // 5
    for index, (_, _, label) in enumerate(GROUPS):
        x0 = margin + index * (width + gap)
        rect = (x0, y, x0 + width, image.height - 35)
        inner = panel(draw, rect, label)
        histogram_rect = (inner[0], inner[1], inner[2], inner[1] + 520)
        frame = query[query["display_group"].eq(label) & query["baseline_eligible"].eq(True)]
        groups = []
        for success, color, name in [(True, COLORS["success"], "success"), (False, COLORS["failure"], "failure")]:
            values = frame.loc[frame["success"].eq(success), "mean_y_L3"].dropna().to_numpy(float)
            if len(values):
                groups.append((values, color, name))
        draw_histogram(draw, histogram_rect, groups, x_label="query mean y (L=3)")
        rank = sensitivity[sensitivity["display_group"].eq(label)]
        draw.text((inner[0] + 10, inner[1] + 555), "Tail-length rank stability", font=FONTS["small"], fill=BLACK)
        for row_index, row in enumerate(rank.itertuples()):
            yy = inner[1] + 605 + row_index * 55
            value = float(row.query_rank_correlation)
            draw.text((inner[0] + 10, yy), f"L{int(row.L_left)} vs L{int(row.L_right)}", font=FONTS["small"], fill=MUTED)
            bar_x0, bar_x1 = inner[0] + 165, inner[2] - 58
            draw.rectangle((bar_x0, yy + 2, bar_x1, yy + 24), fill="#e5e9ee")
            draw.rectangle((bar_x0, yy + 2, int(bar_x0 + value * (bar_x1 - bar_x0)), yy + 24), fill="#5c8db8")
            draw.text((bar_x1 + 8, yy), f"{value:.3f}", font=FONTS["tiny"], fill=BLACK)
    path = output / "01_raw_dvac_and_tail_length.png"
    image.save(path)
    return path


def render_position(horizon: pd.DataFrame, output: Path) -> tuple[Path, pd.DataFrame]:
    image = Image.new("RGB", (2500, 1400), WHITE)
    draw = ImageDraw.Draw(image)
    y = title(draw, image.width, "Signal 2 — Position channel b_h = mu + P_h", "Top: expected log-DVAC by future position. Bottom: the position-specific robust scale s_h used only for normalization.")
    base = add_group_label(horizon[horizon["L"].eq(3)].drop_duplicates(["group_id", "h"]))
    margin, gap = 28, 18
    width = (image.width - 2 * margin - 4 * gap) // 5
    rows: list[dict[str, float | str | int]] = []
    for index, (_, _, label) in enumerate(GROUPS):
        x0 = margin + index * (width + gap)
        rect = (x0, y, x0 + width, image.height - 35)
        inner = panel(draw, rect, label)
        frame = base[base["display_group"].eq(label)].sort_values("h")
        x = frame["h"].to_numpy(float)
        b = frame["b_position"].to_numpy(float)
        scale = frame["mad_scale"].to_numpy(float)
        mu = float(frame["mu"].iloc[0])
        draw_line_chart(
            draw,
            (inner[0], inner[1], inner[2], inner[1] + 510),
            [(x, b, COLORS["position"], "b_h"), (x, np.full_like(x, mu), BLACK, "mu")],
            x_label="future h",
            y_label="raw log unit",
        )
        draw.text((inner[0] + 68, inner[1] + 30), f"P_h is the vertical distance b_h-mu; span={b.max()-b.min():.3f}", font=FONTS["tiny"], fill=MUTED)
        draw_line_chart(
            draw,
            (inner[0], inner[1] + 570, inner[2], inner[1] + 1030),
            [(x, scale, "#93663f", "s_h")],
            x_label="future h",
            y_label="robust scale",
        )
        rows.append(
            {
                "display_group": label,
                "position_span": float(b.max() - b.min()),
                "position_min": float(b.min()),
                "position_max": float(b.max()),
                "mu": mu,
                "scale_min": float(scale.min()),
                "scale_median": float(np.median(scale)),
                "scale_max": float(scale.max()),
                "floored_bins": int(frame["scale_floored"].sum()),
            }
        )
    path = output / "02_position_baseline_and_scale.png"
    image.save(path)
    return path, pd.DataFrame(rows)


def build_residual_episode_metrics(horizon: pd.DataFrame) -> pd.DataFrame:
    main = horizon[horizon["L"].eq(3) & horizon["baseline_eligible"].eq(True)]
    frame = (
        main.groupby(["policy", "task", "group_id", "episode_key", "episode_uid", "success"], as_index=False)
        .agg(
            episode_mean_abs_r_raw=("r_raw", lambda values: float(np.mean(np.abs(values)))),
            episode_mean_abs_R_std=("R_std", lambda values: float(np.mean(np.abs(values)))),
        )
    )
    return add_group_label(frame)


def render_residual(residual_summary: pd.DataFrame, output: Path) -> Path:
    image = Image.new("RGB", (1900, 820), WHITE)
    draw = ImageDraw.Draw(image)
    y = title(draw, image.width, "Signals 3–4 — raw residual r and standardized residual R", "Both remove Position. R additionally divides each h by its own s_h; intervals are descriptive episode bootstraps.")
    draw_interval_panel(image, (35, y, 930, 785), residual_summary, "episode_mean_abs_r_raw", "Mean |r| in raw log units", COLORS["residual"])
    draw_interval_panel(image, (970, y, 1865, 785), residual_summary, "episode_mean_abs_R_std", "Mean |R| after position scaling", COLORS["R"])
    path = output / "03_residual_r_and_R_outcome.png"
    image.save(path)
    return path


def render_four_way(outcome: pd.DataFrame, output: Path) -> Path:
    image = Image.new("RGB", (3000, 850), WHITE)
    draw = ImageDraw.Draw(image)
    y = title(draw, image.width, "Signals 5–6 — query-wide S and local I", "S carries the clearer outcome association here, but its sign reverses on turn-switch. I intervals all cross zero.")
    summary = add_group_label(outcome)
    specs = [
        ("episode_mean_S_raw", "S_raw: query-wide shift", COLORS["S"]),
        ("episode_mean_S_std", "S_std: standardized shift", "#087d5c"),
        ("episode_mean_abs_I_raw", "|I_raw|: local shape", COLORS["I"]),
        ("episode_mean_abs_I_std", "|I_std|: standardized shape", "#a9470b"),
    ]
    margin, gap = 25, 18
    width = (image.width - 2 * margin - 3 * gap) // 4
    for index, (metric, heading, color) in enumerate(specs):
        x0 = margin + index * (width + gap)
        draw_interval_panel(image, (x0, y, x0 + width, 810), summary, metric, heading, color)
    path = output / "04_four_way_S_and_I_outcome.png"
    image.save(path)
    return path


def read_strip(strip_root: Path, policy: str, task: str, episode_uid: str) -> Path:
    return strip_root / CASE_STRIPS[(policy, task, episode_uid)]


def pivot_case(horizon: pd.DataFrame, policy: str, task: str, episode_uid: str, metric: str) -> np.ndarray:
    frame = horizon[
        horizon["policy"].eq(policy)
        & horizon["task"].eq(task)
        & horizon["episode_uid"].eq(episode_uid)
        & horizon["L"].eq(3)
    ]
    return frame.pivot(index="query_idx", columns="h", values=metric).sort_index().sort_index(axis=1).to_numpy(float)


def render_pi0_case(query: pd.DataFrame, horizon: pd.DataFrame, strip_root: Path, output: Path) -> Path:
    image = Image.new("RGB", (2400, 2600), WHITE)
    draw = ImageDraw.Draw(image)
    y = title(
        draw,
        image.width,
        "pi0 adjust-bottle case: remove Position, then split query-wide S from local I",
        "Frames align only to query boundaries. A future h is not claimed to be an individual physics video frame.",
    )
    cases = [("ep000033_reset100100005", "typical success"), ("ep000050_reset100100017", "typical failure")]
    margin, gap = 35, 30
    width = (image.width - 2 * margin - gap) // 2
    for column, (episode_uid, label) in enumerate(cases):
        x0 = margin + column * (width + gap)
        x1 = x0 + width
        draw.rounded_rectangle((x0, y, x1, image.height - 35), radius=16, fill="#fbfcfd", outline="#cfd5dc", width=2)
        draw.text((x0 + 20, y + 14), f"{label}: {episode_uid}", font=FONTS["subtitle"], fill=COLORS["success" if "success" in label else "failure"])
        paste_strip(image, (x0 + 20, y + 58, x1 - 20, y + 310), read_strip(strip_root, "pi0", "adjust_bottle", episode_uid))
        qframe = query[
            query["policy"].eq("pi0") & query["task"].eq("adjust_bottle") & query["episode_uid"].eq(episode_uid)
        ].sort_values("query_idx")
        draw_line_chart(
            draw,
            (x0 + 20, y + 330, x1 - 20, y + 700),
            [
                (qframe["action_slot_start"].to_numpy(float), qframe["S_std_L3"].to_numpy(float), COLORS["S"], "S_std"),
                (qframe["action_slot_start"].to_numpy(float), qframe["mean_abs_I_std_L3"].to_numpy(float), COLORS["I"], "mean |I_std|"),
                (qframe["action_slot_start"].to_numpy(float), qframe["mean_abs_R_std_L3"].to_numpy(float), COLORS["R"], "mean |R|")
            ],
            x_label="query-boundary action slot",
            y_label="std unit",
            zero=True,
        )
        if "success" in label:
            draw.text((x0 + 690, y + 355), "q3 is post-success\n(descriptive only)", font=FONTS["small"], fill="#6f6f6f")
        top = y + 735
        for index, (metric, heading, symmetric) in enumerate(
            [
                ("y_ln", "1  raw log-DVAC y(q,h): Position stripes are still present", False),
                ("r_raw", "3  raw residual r(q,h): subtract b_h", True),
                ("R_std", "4  standardized residual R(q,h): divide by s_h", True),
                ("I_std", "6  local interaction I_std(q,h): subtract query S_std", True),
            ]
        ):
            draw_heatmap(
                image,
                draw,
                (x0 + 20, top + index * 445, x1 - 20, top + index * 445 + 410),
                pivot_case(horizon, "pi0", "adjust_bottle", episode_uid, metric),
                heading=heading,
                symmetric=symmetric,
                gray_last_row=("success" in label),
            )
    path = output / "05_case_pi0_adjust_success_vs_failure.png"
    image.save(path)
    return path


def render_fastwam_case(
    actions: pd.DataFrame,
    strip_root: Path,
    task: str,
    cases: list[tuple[str, str]],
    output: Path,
    filename: str,
) -> Path:
    columns = len(cases)
    image = Image.new("RGB", (1280 if columns == 1 else 2500, 2250), WHITE)
    draw = ImageDraw.Draw(image)
    y = title(
        draw,
        image.width,
        f"Fast-WAM {task.replace('_', ' ')}: action/video-aligned signal decomposition",
        "x = executed physics action slot = fresh pre-action video frame. Only h<24 is shown; h=24..31 was not executed.",
    )
    margin, gap = 35, 30
    width = (image.width - 2 * margin - gap * (columns - 1)) // columns
    for column, (episode_uid, label) in enumerate(cases):
        x0 = margin + column * (width + gap)
        x1 = x0 + width
        draw.rounded_rectangle((x0, y, x1, image.height - 35), radius=16, fill="#fbfcfd", outline="#cfd5dc", width=2)
        draw.text((x0 + 20, y + 14), f"{label}: {episode_uid}", font=FONTS["subtitle"], fill=COLORS["success" if "success" in label else "failure"])
        paste_strip(image, (x0 + 20, y + 58, x1 - 20, y + 325), read_strip(strip_root, "fastwam", task, episode_uid))
        frame = actions[
            actions["policy"].eq("fastwam")
            & actions["task"].eq(task)
            & actions["episode_key"].str.endswith(episode_uid)
            & actions["L"].eq(3)
        ].sort_values("action_slot")
        if frame.empty:
            raise RuntimeError(f"No action metrics for {task} {episode_uid}")
        x = frame["action_slot"].to_numpy(float)
        starts = sorted(frame.loc[frame["h"].eq(0), "action_slot"].astype(int).unique())
        charts = [
            (
                [(x, frame["y_ln"].to_numpy(float), COLORS["raw"], "raw y"), (x, frame["b_position"].to_numpy(float), COLORS["position"], "b_h")],
                "1–2  raw y versus Position b_h",
                "raw log unit",
                False,
            ),
            ([(x, frame["r_raw"].to_numpy(float), COLORS["residual"], "r=y-b")], "3  raw residual r", "raw log unit", True),
            ([(x, frame["R_std"].to_numpy(float), COLORS["R"], "R=r/s_h")], "4  standardized residual R", "std unit", True),
            (
                [
                    (x, frame["S_std"].to_numpy(float), COLORS["S"], "S_std query-wide"),
                    (x, frame["I_std"].to_numpy(float), COLORS["I"], "I_std local"),
                    (x, frame["R_std"].to_numpy(float), "#777777", "R=S+I"),
                ],
                "5–6  split R into query-wide S and local I",
                "std unit",
                True,
            ),
        ]
        top = y + 350
        for index, (series, heading, y_label, zero) in enumerate(charts):
            chart_rect = (x0 + 20, top + index * 430, x1 - 20, top + index * 430 + 400)
            draw.text((chart_rect[0] + 60, chart_rect[1] + 3), heading, font=FONTS["small"], fill=BLACK)
            draw_line_chart(
                draw,
                (chart_rect[0], chart_rect[1] + 28, chart_rect[2], chart_rect[3]),
                series,
                x_label="executed action / video frame",
                y_label=y_label,
                zero=zero,
                verticals=starts,
            )
        if task == "move_stapler_pad":
            draw.text((x0 + 28, image.height - 75), "Move-stapler is the only pair with independent phase annotations; see the teaching text and original storyboards.", font=FONTS["small"], fill=MUTED)
    path = output / filename
    image.save(path)
    return path


def build_signal_summary(
    query: pd.DataFrame,
    position: pd.DataFrame,
    outcome: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    position = position.set_index("display_group")
    outcome = add_group_label(outcome)
    sensitivity = add_group_label(sensitivity)
    rows: list[dict[str, float | int | str]] = []
    for policy, task, label in GROUPS:
        qgroup = query[query["policy"].eq(policy) & query["task"].eq(task)]
        eligible = qgroup[qgroup["baseline_eligible"].eq(True)]
        row: dict[str, float | int | str] = {
            "display_group": label,
            "policy": policy,
            "task": task,
            "episodes": int(qgroup["episode_key"].nunique()),
            "success_episodes": int(qgroup.loc[qgroup["success"].eq(True), "episode_key"].nunique()),
            "failure_episodes": int(qgroup.loc[qgroup["success"].eq(False), "episode_key"].nunique()),
            "queries": int(qgroup["query_key"].nunique()),
            "eligible_queries": int(eligible["query_key"].nunique()),
            "mean_y_L3": float(eligible["mean_y_L3"].mean()),
            **position.loc[label].to_dict(),
            "min_L_rank_correlation": float(sensitivity.loc[sensitivity["display_group"].eq(label), "query_rank_correlation"].min()),
            "max_L_rank_correlation": float(sensitivity.loc[sensitivity["display_group"].eq(label), "query_rank_correlation"].max()),
        }
        for metric in ["episode_mean_S_raw", "episode_mean_S_std", "episode_mean_abs_I_raw", "episode_mean_abs_I_std", "episode_mean_abs_R_std"]:
            match = outcome[outcome["display_group"].eq(label) & outcome["metric"].eq(metric)]
            if not match.empty:
                row[f"{metric}_success_minus_failure"] = float(match["success_minus_failure"].iloc[0])
                row[f"{metric}_ci_low"] = float(match["bootstrap_ci_low"].iloc[0])
                row[f"{metric}_ci_high"] = float(match["bootstrap_ci_high"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.analysis_root.resolve()
    output = args.output.resolve()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    query = pd.read_csv(root / "query_metrics.csv", low_memory=False)
    horizon = pd.read_csv(root / "query_horizon.csv", low_memory=False)
    outcome = pd.read_csv(root / "outcome_summary.csv")
    sensitivity = pd.read_csv(root / "L_sensitivity.csv")
    actions = pd.read_csv(root / "fastwam_action_frame_metrics.csv", low_memory=False)
    strip_root = root / "query_frame_strips"

    created = [render_signal_map(figures), render_raw(query, sensitivity, figures)]
    position_path, position_summary = render_position(horizon, figures)
    created.append(position_path)
    residual_episode = build_residual_episode_metrics(horizon)
    residual_summary = summarize_differences(residual_episode, ["episode_mean_abs_r_raw", "episode_mean_abs_R_std"])
    created.extend([render_residual(residual_summary, figures), render_four_way(outcome, figures)])
    created.append(render_pi0_case(query, horizon, strip_root, figures))
    created.append(render_fastwam_case(actions, strip_root, "adjust_bottle", [("episode0002_reset1", "typical success; all 16 succeeded")], figures, "06_case_fastwam_adjust_success.png"))
    created.append(render_fastwam_case(actions, strip_root, "move_stapler_pad", [("episode0004_reset3", "typical success"), ("episode0006_reset5", "typical failure")], figures, "07_case_fastwam_move_success_vs_failure.png"))
    created.append(render_fastwam_case(actions, strip_root, "turn_switch", [("episode0009_reset8", "typical success"), ("episode0001_reset0", "typical failure")], figures, "08_case_fastwam_turn_success_vs_failure.png"))
    created.append(render_fastwam_case(actions, strip_root, "pick_diverse_bottles", [("episode0004_reset3", "typical success"), ("episode0005_reset4", "typical failure")], figures, "09_case_fastwam_pick_success_vs_failure.png"))

    position_summary.to_csv(output / "position_scale_summary.csv", index=False)
    residual_episode.to_csv(output / "residual_episode_metrics.csv", index=False)
    residual_summary.to_csv(output / "residual_outcome_summary.csv", index=False)
    build_signal_summary(query, position_summary, outcome, sensitivity).to_csv(output / "signal_summary.csv", index=False)
    pd.DataFrame({"figure": [str(path) for path in created]}).to_csv(output / "figure_index.csv", index=False)
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
