"""Offline counterfactuals for the Step-39 DVAC per-h gradient weights.

This reads the frozen rollout tensors only.  It does not touch the running
training process or recompute policy gradients.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DVAC_ROOT = ROOT / "run" / "dvac_train"
OUT = Path(__file__).resolve().parent
H = 50
EPS = 1e-12


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)


def read_rank_history(rank_dir: Path) -> dict[int, tuple[float, float]]:
    with (rank_dir / "runner_step_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    return {
        int(row["runner_step"]):
        (float(row["history_mean"]), float(row["history_std"]))
        for row in rows
    }


def load_clipped_z() -> tuple[np.ndarray, np.ndarray]:
    chunks: list[np.ndarray] = []
    steps: list[np.ndarray] = []
    max_formula_error = 0.0
    for rank_dir in sorted(DVAC_ROOT.glob("actor_rank*")):
        history = read_rank_history(rank_dir)
        for path in sorted(rank_dir.glob("rollout_step*.npz")):
            match = re.search(r"rollout_step(\d+)\.npz$", path.name)
            if match is None:
                continue
            runner_step = int(match.group(1))
            if runner_step == 0:
                continue
            mean, std = history[runner_step]
            with np.load(path) as data:
                variance = data["v_l3"].astype(np.float64).reshape(-1, H)
                valid = data["loss_mask"].astype(bool).reshape(-1)
                stored_weights = data["weights"].astype(np.float64).reshape(-1, H)
                variance = variance[valid]
                stored_weights = stored_weights[valid]
            z = (np.log(variance + EPS) - mean) / max(std, 1e-6)
            clipped = np.clip(z, -2.0, 2.0)
            max_formula_error = max(
                max_formula_error,
                float(np.max(np.abs(stored_weights - (1.0 + 0.1 * clipped)))),
            )
            chunks.append(clipped.astype(np.float32))
            steps.append(np.full(clipped.shape[0], runner_step, dtype=np.int16))
    clipped_z = np.concatenate(chunks, axis=0)
    runner_steps = np.concatenate(steps, axis=0)
    if max_formula_error > 1e-5:
        raise RuntimeError(f"stored weight formula mismatch: {max_formula_error}")
    print(f"valid_queries={clipped_z.shape[0]} formula_error={max_formula_error:.3g}")
    return clipped_z, runner_steps


def top20_mask(signal: np.ndarray) -> np.ndarray:
    top_k = int(round(signal.shape[1] * 0.2))
    indices = np.argpartition(signal, -top_k, axis=1)[:, -top_k:]
    mask = np.zeros_like(signal, dtype=bool)
    np.put_along_axis(mask, indices, True, axis=1)
    return mask


def make_schemes(signal: np.ndarray) -> dict[str, np.ndarray]:
    schemes: dict[str, np.ndarray] = {}
    for strength in (0.1, 0.2, 0.3):
        raw = 1.0 + strength * signal
        schemes[f"symmetric_a{strength:.1f}"] = raw
        schemes[f"mean1_symmetric_a{strength:.1f}"] = raw / raw.mean(
            axis=1, keepdims=True
        )
        up = 1.0 + strength * np.maximum(signal, 0.0)
        schemes[f"up_only_a{strength:.1f}"] = up
        schemes[f"mean1_up_only_a{strength:.1f}"] = up / up.mean(
            axis=1, keepdims=True
        )

    selected = top20_mask(signal)
    # A bounded hard selector with exactly mean-one weight per query:
    # 20% get 1.2 and the other 80% get 0.95.
    schemes["bounded_hard_top20_mean1"] = np.where(selected, 1.2, 0.95)
    # Beyond-80/20-style hard masking before and after selected-count
    # normalization.  Both have the same direction whenever global clipping
    # is active; they differ only in pre-clip magnitude.
    schemes["hardmask_top20_raw"] = selected.astype(np.float64)
    schemes["hardmask_top20_mean1"] = selected.astype(np.float64) * 5.0
    return schemes


def summarize(scope: str, name: str, weights: np.ndarray) -> dict[str, float | int | str]:
    flat = weights.reshape(-1)
    qsum = weights.sum(axis=1)
    qsq = np.square(weights).sum(axis=1)
    q_ess = np.divide(
        np.square(qsum),
        H * qsq,
        out=np.zeros_like(qsum),
        where=qsq > 0,
    )
    total_sq = float(np.square(flat).sum())
    ess_ratio = float(flat.sum() ** 2 / (flat.size * total_sq)) if total_sq else 0.0
    cosine_proxy = math.sqrt(max(ess_ratio, 0.0))
    angle_proxy = math.degrees(math.acos(min(max(cosine_proxy, -1.0), 1.0)))
    p = np.quantile(flat, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "scope": scope,
        "scheme": name,
        "valid_queries": int(weights.shape[0]),
        "positions": int(flat.size),
        "weight_min": float(flat.min()),
        "weight_p01": float(p[0]),
        "weight_p05": float(p[1]),
        "weight_p25": float(p[2]),
        "weight_median": float(p[3]),
        "weight_mean": float(flat.mean()),
        "weight_p75": float(p[4]),
        "weight_p95": float(p[5]),
        "weight_p99": float(p[6]),
        "weight_max": float(flat.max()),
        "weight_std": float(flat.std()),
        "mean_square_weight": float(np.square(flat).mean()),
        "l2_scale_proxy": float(np.sqrt(np.square(flat).mean())),
        "global_ess_fraction": ess_ratio,
        "query_ess_fraction_p05": float(np.quantile(q_ess, 0.05)),
        "query_ess_fraction_median": float(np.median(q_ess)),
        "query_ess_fraction_mean": float(q_ess.mean()),
        "orthogonal_equal_gradient_cosine_proxy": cosine_proxy,
        "orthogonal_equal_gradient_angle_deg_proxy": angle_proxy,
        "fraction_zero": float(np.mean(flat == 0.0)),
        "fraction_below_one": float(np.mean(flat < 1.0)),
        "fraction_above_one": float(np.mean(flat > 1.0)),
        "query_mean_p05": float(np.quantile(weights.mean(axis=1), 0.05)),
        "query_mean_median": float(np.median(weights.mean(axis=1))),
        "query_mean_p95": float(np.quantile(weights.mean(axis=1), 0.95)),
        "front_h00_24_mean": float(weights[:, :25].mean()),
        "back_h25_49_mean": float(weights[:, 25:].mean()),
        "back_minus_front": float(weights[:, 25:].mean() - weights[:, :25].mean()),
        "back_over_front": float(weights[:, 25:].mean() / weights[:, :25].mean())
        if weights[:, :25].mean() != 0
        else math.inf,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_plot(
    summary_rows: list[dict[str, object]],
    by_h_rows: list[dict[str, object]],
) -> None:
    width, height = 2200, 1320
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title = load_font(45, True)
    subtitle = load_font(26)
    label = load_font(24)
    small = load_font(20)
    tiny = load_font(18)
    draw.text((70, 40), "DVAC per-h gradient-weight counterfactuals (through Global Step 39)", fill="#111827", font=title)
    draw.text((70, 100), "Offline replay of recorded V_L3 only; PPO gradients were not recomputed", fill="#4b5563", font=subtitle)

    all_rows = {str(row["scheme"]): row for row in summary_rows if row["scope"] == "all_apply_g2_g39"}
    latest_rows = {str(row["scheme"]): row for row in summary_rows if row["scope"] == "latest_g39"}
    continuous = [
        "symmetric_a0.1",
        "symmetric_a0.2",
        "symmetric_a0.3",
        "mean1_symmetric_a0.3",
        "up_only_a0.3",
        "mean1_up_only_a0.3",
        "bounded_hard_top20_mean1",
    ]
    names = {
        "symmetric_a0.1": "current sym a=.1",
        "symmetric_a0.2": "sym a=.2",
        "symmetric_a0.3": "sym a=.3",
        "mean1_symmetric_a0.3": "mean-1 sym a=.3",
        "up_only_a0.3": "up-only a=.3",
        "mean1_up_only_a0.3": "mean-1 up a=.3",
        "bounded_hard_top20_mean1": "bounded hard top20",
        "hardmask_top20_raw": "hard mask raw {0,1}",
        "hardmask_top20_mean1": "hard mask mean-1 {0,5}",
    }

    # Panel A: all-apply weight distributions.
    x0, y0, x1, y1 = 70, 180, 1070, 720
    draw.rounded_rectangle((x0, y0, x1, y1), radius=18, fill="#f8fafc", outline="#d1d5db", width=2)
    draw.text((x0 + 25, y0 + 20), "A. Continuous / bounded alternatives: all apply steps", fill="#111827", font=label)
    axis_left, axis_right = x0 + 260, x1 - 40
    axis_top, axis_bottom = y0 + 90, y1 - 45
    xmin, xmax = 0.2, 2.5
    for tick in np.arange(0.5, 2.51, 0.5):
        px = axis_left + (tick - xmin) / (xmax - xmin) * (axis_right - axis_left)
        draw.line((px, axis_top, px, axis_bottom), fill="#e5e7eb", width=1)
        draw.text((px - 18, axis_bottom + 8), f"{tick:.1f}", fill="#6b7280", font=tiny)
    row_h = (axis_bottom - axis_top) / len(continuous)
    for idx, key in enumerate(continuous):
        row = all_rows[key]
        cy = axis_top + (idx + 0.5) * row_h
        draw.text((x0 + 25, cy - 12), names[key], fill="#374151", font=tiny)
        def xp(value: float) -> float:
            return axis_left + (value - xmin) / (xmax - xmin) * (axis_right - axis_left)
        draw.line((xp(float(row["weight_min"])), cy, xp(float(row["weight_max"])), cy), fill="#94a3b8", width=3)
        draw.line((xp(float(row["weight_p05"])), cy, xp(float(row["weight_p95"])), cy), fill="#2563eb", width=10)
        mx = xp(float(row["weight_median"]))
        draw.ellipse((mx - 7, cy - 7, mx + 7, cy + 7), fill="#f97316")
    draw.text((axis_left, y1 - 22), "line=min..max, blue=p05..p95, orange=median", fill="#6b7280", font=tiny)

    # Panel B: ESS and equal-orthogonal direction proxy.
    x0, y0, x1, y1 = 1130, 180, 2130, 720
    draw.rounded_rectangle((x0, y0, x1, y1), radius=18, fill="#f8fafc", outline="#d1d5db", width=2)
    draw.text((x0 + 25, y0 + 20), "B. Weight concentration (ESS fraction)", fill="#111827", font=label)
    shown = continuous + ["hardmask_top20_raw", "hardmask_top20_mean1"]
    axis_left, axis_right = x0 + 290, x1 - 55
    axis_top, axis_bottom = y0 + 90, y1 - 55
    for tick in np.arange(0.0, 1.01, 0.2):
        px = axis_left + tick * (axis_right - axis_left)
        draw.line((px, axis_top, px, axis_bottom), fill="#e5e7eb", width=1)
        draw.text((px - 16, axis_bottom + 9), f"{tick:.1f}", fill="#6b7280", font=tiny)
    row_h = (axis_bottom - axis_top) / len(shown)
    for idx, key in enumerate(shown):
        row = all_rows[key]
        cy = axis_top + idx * row_h + 4
        draw.text((x0 + 25, cy - 3), names[key], fill="#374151", font=tiny)
        ess = float(row["global_ess_fraction"])
        draw.rectangle((axis_left, cy, axis_left + ess * (axis_right - axis_left), cy + 17), fill="#0ea5e9")
        draw.text((axis_left + ess * (axis_right - axis_left) + 8, cy - 4), f"{ess:.3f}", fill="#111827", font=tiny)
    draw.text((x0 + 25, y1 - 27), "1.0 = uniform weights; 0.2 = only 20% retained", fill="#6b7280", font=tiny)

    # Panel C: average h profile.
    x0, y0, x1, y1 = 70, 770, 1560, 1260
    draw.rounded_rectangle((x0, y0, x1, y1), radius=18, fill="#f8fafc", outline="#d1d5db", width=2)
    draw.text((x0 + 25, y0 + 20), "C. Mean weight by future-action index h", fill="#111827", font=label)
    axis_left, axis_right = x0 + 85, x1 - 35
    axis_top, axis_bottom = y0 + 75, y1 - 75
    ymin, ymax = 0.8, 1.35
    for tick in np.arange(0.8, 1.31, 0.1):
        py = axis_bottom - (tick - ymin) / (ymax - ymin) * (axis_bottom - axis_top)
        draw.line((axis_left, py, axis_right, py), fill="#e5e7eb", width=1)
        draw.text((x0 + 22, py - 10), f"{tick:.1f}", fill="#6b7280", font=tiny)
    for tick in (0, 10, 20, 30, 40, 49):
        px = axis_left + tick / 49 * (axis_right - axis_left)
        draw.text((px - 9, axis_bottom + 10), str(tick), fill="#6b7280", font=tiny)
    right_ymin, right_ymax = 0.0, 3.2
    for tick in (0.0, 1.0, 2.0, 3.0):
        py = axis_bottom - (tick - right_ymin) / (right_ymax - right_ymin) * (axis_bottom - axis_top)
        draw.text((axis_right + 5, py - 10), f"{tick:.0f}", fill="#dc2626", font=tiny)
    colors = {
        "symmetric_a0.1": "#2563eb",
        "symmetric_a0.2": "#16a34a",
        "symmetric_a0.3": "#f97316",
        "bounded_hard_top20_mean1": "#7c3aed",
        "hardmask_top20_mean1": "#dc2626",
    }
    by_key: dict[str, list[dict[str, object]]] = {}
    for row in by_h_rows:
        if row["scope"] == "all_apply_g2_g39":
            by_key.setdefault(str(row["scheme"]), []).append(row)
    for key, color in colors.items():
        rows = sorted(by_key[key], key=lambda item: int(item["h"]))
        points = []
        for row in rows:
            px = axis_left + int(row["h"]) / 49 * (axis_right - axis_left)
            value = float(row["weight_mean"])
            if key == "hardmask_top20_mean1":
                py = axis_bottom - (value - right_ymin) / (right_ymax - right_ymin) * (axis_bottom - axis_top)
            else:
                py = axis_bottom - (value - ymin) / (ymax - ymin) * (axis_bottom - axis_top)
            points.append((px, py))
        draw.line(points, fill=color, width=4)
    lx, ly = axis_left + 10, axis_top + 10
    for key, color in colors.items():
        draw.line((lx, ly + 10, lx + 35, ly + 10), fill=color, width=5)
        legend_name = names[key] + (" (right axis)" if key == "hardmask_top20_mean1" else "")
        draw.text((lx + 45, ly), legend_name, fill="#374151", font=tiny)
        lx += 265
        if lx > axis_right - 220:
            lx = axis_left + 10
            ly += 30
    draw.text((axis_left, y1 - 28), "Higher h is systematically higher-V in this run, so all monotone schemes emphasize the chunk tail", fill="#6b7280", font=tiny)

    # Panel D: concise latest/all facts.
    x0, y0, x1, y1 = 1620, 770, 2130, 1260
    draw.rounded_rectangle((x0, y0, x1, y1), radius=18, fill="#eff6ff", outline="#93c5fd", width=2)
    draw.text((x0 + 25, y0 + 20), "D. Readout", fill="#111827", font=label)
    current_all = all_rows["symmetric_a0.1"]
    current_latest = latest_rows["symmetric_a0.1"]
    hard = all_rows["hardmask_top20_mean1"]
    lines = [
        f"Current all-step mean: {float(current_all['weight_mean']):.3f}",
        f"Current latest mean:   {float(current_latest['weight_mean']):.3f}",
        f"Current ESS retained:  {float(current_all['global_ess_fraction']):.3f}",
        f"Current back-front:    {float(current_all['back_minus_front']):+.3f}",
        f"Sym a=.3 ESS:          {float(all_rows['symmetric_a0.3']['global_ess_fraction']):.3f}",
        f"Hard 20% ESS:          {float(hard['global_ess_fraction']):.3f}",
        f"Hard 20% zeros:        {100*float(hard['fraction_zero']):.1f}%",
        "",
        "ESS is a weight-concentration diagnostic.",
        "It is not a measured policy-gradient ESS.",
        "The angle proxy assumes equal, orthogonal",
        "per-h gradient vectors; real gradients were",
        "not saved, so no actual gradient angle is claimed.",
    ]
    yy = y0 + 80
    for line in lines:
        draw.text((x0 + 25, yy), line, fill="#1f2937", font=small)
        yy += 31

    image.save(OUT / "DVAC_WEIGHT_COUNTERFACTUAL.png", optimize=True)


def main() -> None:
    clipped_z, runner_steps = load_clipped_z()
    latest_step = int(runner_steps.max())
    scopes = {
        "all_apply_g2_g39": np.ones(runner_steps.shape, dtype=bool),
        "latest_g39": runner_steps == latest_step,
    }
    summary_rows: list[dict[str, object]] = []
    by_h_rows: list[dict[str, object]] = []
    for scope, mask in scopes.items():
        signal = clipped_z[mask]
        schemes = make_schemes(signal)
        for name, weights in schemes.items():
            summary_rows.append(summarize(scope, name, weights))
            for h in range(H):
                column = weights[:, h]
                by_h_rows.append(
                    {
                        "scope": scope,
                        "scheme": name,
                        "h": h,
                        "valid_queries": int(weights.shape[0]),
                        "weight_mean": float(column.mean()),
                        "weight_p05": float(np.quantile(column, 0.05)),
                        "weight_median": float(np.median(column)),
                        "weight_p95": float(np.quantile(column, 0.95)),
                        "fraction_zero": float(np.mean(column == 0.0)),
                        "fraction_above_one": float(np.mean(column > 1.0)),
                    }
                )

    write_csv(OUT / "DVAC_WEIGHT_COUNTERFACTUAL.csv", summary_rows)
    write_csv(OUT / "DVAC_WEIGHT_COUNTERFACTUAL_BY_H.csv", by_h_rows)
    payload = {
        "source": str(DVAC_ROOT),
        "latest_runner_step_internal": latest_step,
        "latest_global_step": latest_step + 1,
        "valid_queries_all_apply": int(clipped_z.shape[0]),
        "valid_queries_latest": int(np.sum(runner_steps == latest_step)),
        "note": (
            "Counterfactual weights replay recorded clipped z only. ESS and "
            "equal-orthogonal angle are diagnostics, not recomputed gradients."
        ),
        "rows": summary_rows,
    }
    (OUT / "DVAC_WEIGHT_COUNTERFACTUAL_SUMMARY.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    draw_plot(summary_rows, by_h_rows)


if __name__ == "__main__":
    main()
