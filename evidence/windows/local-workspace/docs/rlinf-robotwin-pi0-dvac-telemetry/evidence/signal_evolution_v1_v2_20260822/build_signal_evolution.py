from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
V1_ROOT = ROOT / "exports/idea2_dvac_v1_formal_stop_g54_20260821/run/dvac_train"
V2_STEP = (
    ROOT
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/"
    "r_only_formal_live_g35_20260822/analysis/DVAC_STEP_METRICS_G35.csv"
)
V2_RAW = (
    ROOT
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/"
    "r_only_formal_live_g35_20260822/raw/run/dvac_train"
)


def number(value: str | float | None) -> float:
    if value in (None, "", "nan"):
        return float("nan")
    return float(value)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate_v1() -> list[dict[str, float]]:
    ranks = [read_csv(V1_ROOT / f"actor_rank0{rank}/runner_step_metrics.csv") for rank in (0, 1)]
    result: list[dict[str, float]] = []
    weighted = [
        "weight_p05", "weight_mean", "weight_p50", "weight_p95",
        "z_low_clip_fraction", "z_high_clip_fraction",
        "positive_adv_weight_mean", "negative_adv_weight_mean",
    ]
    for index, (left, right) in enumerate(zip(*ranks)):
        counts = np.asarray([number(left["valid_queries_local"]), number(right["valid_queries_local"])])
        denom = float(counts.sum())
        row = {
            "step": float(index + 1),
            "current_mean": number(left["current_mean"]),
            "current_std": number(left["current_std"]),
            "history_mean": number(left["history_mean"]) if index else float("nan"),
            "history_std": number(left["history_std"]) if index else float("nan"),
        }
        for key in weighted:
            values = np.asarray([number(left[key]), number(right[key])])
            row[key] = float(np.dot(values, counts) / denom) if denom else float(np.mean(values))
        result.append(row)
    return result


def read_v2() -> list[dict[str, float]]:
    rows = []
    for raw in read_csv(V2_STEP):
        row = {key: number(value) for key, value in raw.items() if key != "signal_mode"}
        if row.get("warmup") == 1:
            row["history_mean"] = float("nan")
            row["history_std"] = float("nan")
        rows.append(row)
    return rows


v1 = aggregate_v1()
v2 = read_v2()


def arr(rows: list[dict[str, float]], key: str) -> np.ndarray:
    return np.asarray([row.get(key, float("nan")) for row in rows], dtype=float)


def latest_v2_components() -> dict[str, object]:
    shards = [np.load(V2_RAW / f"actor_rank0{rank}/rollout_step0034.npz") for rank in (0, 1)]
    residual = np.concatenate([z["residual_z"].reshape(-1, 50) for z in shards])
    valid = np.concatenate([z["loss_mask"].reshape(-1).astype(bool) for z in shards])
    residual = residual[valid]
    state = residual.mean(axis=1)
    local = residual - state[:, None]
    center = shards[0]["position_center_log_v"].astype(float)
    position = center - center.mean()
    return {
        "global_step": 35,
        "valid_queries": int(valid.sum()),
        "mu_position_center_mean_log_v": float(center.mean()),
        "position_effect_std_log_v": float(position.std()),
        "position_back25_minus_front25_log_v": float(center[25:].mean() - center[:25].mean()),
        "state_effect_p05_p50_p95": np.quantile(state, [0.05, 0.5, 0.95]).tolist(),
        "local_effect_p05_p50_p95": np.quantile(local, [0.05, 0.5, 0.95]).tolist(),
        "residual_variance": float(residual.var()),
        "state_variance_share": float(state.var() / residual.var()),
        "local_variance_share": float(local.var() / residual.var()),
    }


components = latest_v2_components()


FONT = Path(r"C:\Windows\Fonts\segoeui.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\seguisb.ttf")
INK, MUTED, GRID, WHITE = "#17233D", "#5E6C84", "#D9E1EA", "#FFFFFF"
ORANGE, PURPLE, BLUE, GREEN, RED = "#E87500", "#7C3AED", "#2563EB", "#0F8A62", "#D33F49"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT), size=size)


def draw_plot(draw, box, title, series, ylabel, y_limits=None, zero=None):
    left, top, right, bottom = box
    draw.text((left, top), title, fill=INK, font=font(25, True))
    pl, pr, pt, pb = left + 88, right - 20, top + 54, bottom - 44
    finite = np.concatenate([ys[np.isfinite(ys)] for _, _, ys, _, _ in series])
    low, high = (float(finite.min()), float(finite.max())) if y_limits is None else y_limits
    if y_limits is None:
        pad = 0.12 * (high - low) if high > low else 1.0
        low, high = low - pad, high + pad
    max_x = max(float(np.nanmax(xs)) for _, xs, _, _, _ in series)

    def px(x): return pl + (x - 1) / max(max_x - 1, 1) * (pr - pl)
    def py(y): return pb - (y - low) / (high - low) * (pb - pt)

    for tick in range(5):
        value = low + (high - low) * tick / 4
        y = py(value)
        draw.line((pl, y, pr, y), fill=GRID, width=2)
        label = f"{value:.2f}" if abs(value) < 10 else f"{value:.0f}"
        width = draw.textlength(label, font=font(15))
        draw.text((pl - width - 10, y - 9), label, fill=MUTED, font=font(15))
    if zero is not None and low <= zero <= high:
        y = py(zero)
        for x in range(int(pl), int(pr), 18):
            draw.line((x, y, min(x + 9, pr), y), fill=MUTED, width=2)
    draw.line((pl, pt, pl, pb), fill=INK, width=2)
    draw.line((pl, pb, pr, pb), fill=INK, width=2)
    draw.text((left, pt), ylabel, fill=MUTED, font=font(15))
    for tick in range(5):
        xval = 1 + (max_x - 1) * tick / 4
        x = px(xval)
        text = str(int(round(xval)))
        draw.text((x - draw.textlength(text, font=font(15)) / 2, pb + 9), text, fill=MUTED, font=font(15))

    lx, ly = pl, pt + 3
    for label, xs, ys, color, dashed in series:
        points = [(px(float(x)), py(float(y))) for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y)]
        if dashed:
            for i in range(len(points) - 1):
                if i % 2 == 0:
                    draw.line((points[i], points[i + 1]), fill=color, width=3)
        elif len(points) > 1:
            draw.line(points, fill=color, width=4, joint="curve")
        if points:
            x, y = points[-1]
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        draw.line((lx, ly + 9, lx + 24, ly + 9), fill=color, width=4)
        draw.text((lx + 31, ly), label, fill=INK, font=font(14))
        lx += int(draw.textlength(label, font=font(14))) + 67


steps1, steps2 = arr(v1, "step"), arr(v2, "global_step")
img = Image.new("RGB", (2100, 1580), WHITE)
draw = ImageDraw.Draw(img)
draw.text((60, 38), "DVAC signal evolution: v1 global-z vs v2 position-residual", fill=INK, font=font(40, True))
draw.text((62, 96), "On-policy train-SDE telemetry. v1 through g54; v2 live snapshot through g35.", fill=MUTED, font=font(20))

draw_plot(draw, (45, 155, 1030, 580), "A. Raw signal level", [
    ("v1 current", steps1, arr(v1, "current_mean"), ORANGE, False),
    ("v1 recent history", steps1, arr(v1, "history_mean"), ORANGE, True),
    ("v2 current", steps2, arr(v2, "current_mean"), PURPLE, False),
    ("v2 recent history", steps2, arr(v2, "history_mean"), PURPLE, True),
], "mean ln(V_L3+eps)")

relative1 = 100 * (np.exp(arr(v1, "current_mean") - arr(v1, "current_mean")[0]) - 1)
relative2 = 100 * (np.exp(arr(v2, "current_mean") - arr(v2, "current_mean")[0]) - 1)
draw_plot(draw, (1070, 155, 2055, 580), "B. Raw V relative to each run's g1", [
    ("v1", steps1, relative1, ORANGE, False),
    ("v2", steps2, relative2, PURPLE, False),
], "geometric mean change %", zero=0)

z_proxy = (arr(v1, "current_mean") - arr(v1, "history_mean")) / arr(v1, "history_std")
draw_plot(draw, (45, 620, 1030, 1045), "C. v1 actual global normalization", [
    ("global mean z proxy", steps1, z_proxy, BLUE, False),
    ("low-cap fraction x10", steps1, 10 * arr(v1, "z_low_clip_fraction"), GREEN, True),
    ("high-cap fraction x10", steps1, 10 * arr(v1, "z_high_clip_fraction"), RED, True),
], "standardized / scaled", zero=0)

draw_plot(draw, (1070, 620, 2055, 1045), "D. v2 position baseline level", [
    ("current raw mean", steps2, arr(v2, "current_mean"), PURPLE, False),
    ("median b_h", steps2, arr(v2, "position_center_median"), BLUE, False),
    ("history raw mean", steps2, arr(v2, "history_mean"), MUTED, True),
], "ln(V_L3+eps)")

draw_plot(draw, (45, 1080, 1030, 1515), "E. v2 actual residual after per-h calibration", [
    ("p05", steps2, arr(v2, "residual_p05"), BLUE, False),
    ("median", steps2, arr(v2, "residual_p50"), PURPLE, False),
    ("p95", steps2, arr(v2, "residual_p95"), ORANGE, False),
], "robust residual (clipped)", y_limits=(-2.2, 2.2), zero=0)

draw_plot(draw, (1070, 1080, 2055, 1515), "F. Output action-h gradient weights", [
    ("v1 p05", steps1, arr(v1, "weight_p05"), ORANGE, True),
    ("v1 median", steps1, arr(v1, "weight_p50"), ORANGE, False),
    ("v1 p95", steps1, arr(v1, "weight_p95"), RED, True),
    ("v2 p05", steps2, arr(v2, "weight_p05"), PURPLE, True),
    ("v2 median", steps2, arr(v2, "weight_p50"), PURPLE, False),
    ("v2 p95", steps2, arr(v2, "weight_p95"), BLUE, True),
], "weight", y_limits=(0.45, 1.25), zero=1.0)

img.save(HERE / "DVAC_SIGNAL_EVOLUTION_V1_G54_V2_G35.png", optimize=True)


summary = {
    "v1": {
        "latest_global_step": int(steps1[-1]),
        "raw_mean_log_v_start_end": [float(arr(v1, "current_mean")[0]), float(arr(v1, "current_mean")[-1])],
        "raw_geometric_mean_v_change_pct": float(relative1[-1]),
        "actual_signal": "global z-score over all q,h; fixed future-h trend retained",
    },
    "v2": {
        "latest_global_step": int(steps2[-1]),
        "raw_mean_log_v_start_end": [float(arr(v2, "current_mean")[0]), float(arr(v2, "current_mean")[-1])],
        "raw_geometric_mean_v_change_pct": float(relative2[-1]),
        "mean_residual_median_g2_latest": float(np.nanmean(arr(v2, "residual_p50")[1:])),
        "latest_residual_median": float(arr(v2, "residual_p50")[-1]),
        "actual_signal": "recent-5 per-h median/MAD residual",
        "latest_four_channel_decomposition": components,
    },
    "boundary": (
        "Raw log-V is an on-policy endpoint-revision statistic, not Shannon entropy. "
        "Its step trend mixes model change with the changing visited-state distribution."
    ),
}
(HERE / "SIGNAL_EVOLUTION_SUMMARY.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)

with (HERE / "SIGNAL_EVOLUTION.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    fields = [
        "run", "global_step", "current_mean_log_v", "history_mean_log_v",
        "position_center_median_log_v", "residual_p05", "residual_p50", "residual_p95",
        "weight_p05", "weight_p50", "weight_p95", "positive_adv_weight_mean", "negative_adv_weight_mean",
    ]
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for label, rows, step_key in (("v1_global_z", v1, "step"), ("v2_r_only", v2, "global_step")):
        for row in rows:
            writer.writerow({
                "run": label,
                "global_step": int(row[step_key]),
                "current_mean_log_v": row.get("current_mean"),
                "history_mean_log_v": row.get("history_mean"),
                "position_center_median_log_v": row.get("position_center_median"),
                "residual_p05": row.get("residual_p05"),
                "residual_p50": row.get("residual_p50"),
                "residual_p95": row.get("residual_p95"),
                "weight_p05": row.get("weight_p05"),
                "weight_p50": row.get("weight_p50"),
                "weight_p95": row.get("weight_p95"),
                "positive_adv_weight_mean": row.get("positive_adv_weight_mean"),
                "negative_adv_weight_mean": row.get("negative_adv_weight_mean"),
            })

print(json.dumps(summary, indent=2, ensure_ascii=False))
