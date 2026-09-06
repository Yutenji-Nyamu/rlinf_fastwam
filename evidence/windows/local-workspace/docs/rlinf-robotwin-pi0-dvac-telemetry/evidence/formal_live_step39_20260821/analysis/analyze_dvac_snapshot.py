from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE.parent
DVAC_DIR = SNAPSHOT / "run" / "dvac_train"
CONTROL_DIR = (
    SNAPSHOT
    / "run"
    / "control_trace"
    / "adjust_bottle"
    / "worker_000"
    / "env_slot_000"
    / "recording_0000_episode_0000_reset_57"
)

EPS = 1e-12
CLIP_LOW = 0.8
CLIP_HIGH = 1.2


def q(values: np.ndarray, p: float) -> float:
    return float(np.quantile(values, p))


def spearman_h(values: np.ndarray) -> float:
    ranks = pd.Series(values).rank(method="average").to_numpy(dtype=np.float64)
    return float(np.corrcoef(np.arange(values.size, dtype=np.float64), ranks)[0, 1])


def pooled_mean_std(arrays: list[np.ndarray]) -> tuple[int, float, float]:
    values = np.concatenate([a.reshape(-1).astype(np.float64) for a in arrays])
    return values.size, float(values.mean()), float(values.std(ddof=0))


def load_inputs():
    rank_dirs = sorted(DVAC_DIR.glob("actor_rank*"))
    if len(rank_dirs) != 2:
        raise RuntimeError(f"Expected two actor ranks, found {len(rank_dirs)}")

    rank_csv = {}
    for rank_dir in rank_dirs:
        rank = int(rank_dir.name[-2:])
        frame = pd.read_csv(rank_dir / "runner_step_metrics.csv")
        frame = frame.set_index("runner_step", drop=False)
        rank_csv[rank] = frame

    step_sets = []
    for rank_dir in rank_dirs:
        steps = {
            int(p.stem.removeprefix("rollout_step"))
            for p in rank_dir.glob("rollout_step*.npz")
        }
        step_sets.append(steps)
    steps = sorted(set.intersection(*step_sets))
    if steps != list(range(steps[-1] + 1)):
        raise RuntimeError(f"Paired NPZ steps are not contiguous: {steps}")

    return rank_dirs, rank_csv, steps


def analyze():
    rank_dirs, rank_csv, steps = load_inputs()
    step_rows: list[dict] = []
    h_rows: list[dict] = []
    step_cache: dict[int, dict[str, np.ndarray]] = {}
    current_log_by_step: dict[int, np.ndarray] = {}

    global_csv_fields = [
        "warmup",
        "history_steps",
        "history_count",
        "history_mean",
        "history_std",
        "current_count",
        "current_mean",
        "current_std",
        "actor_grad_norm",
        "actor_clip_fraction",
        "actor_approx_kl",
        "actor_ratio",
        "actor_policy_loss",
    ]

    for step in steps:
        chunks: dict[str, list[np.ndarray]] = {}
        csv_rows = []
        for rank_dir in rank_dirs:
            rank = int(rank_dir.name[-2:])
            path = rank_dir / f"rollout_step{step:04d}.npz"
            with np.load(path, allow_pickle=False) as data:
                for key in data.files:
                    chunks.setdefault(key, []).append(np.asarray(data[key]))
            csv_rows.append(rank_csv[rank].loc[step])

        combined = {}
        for key, arrays in chunks.items():
            if arrays[0].ndim >= 2 and arrays[0].shape[:2] == (4, 128):
                combined[key] = np.concatenate(
                    [a.reshape((-1,) + a.shape[2:]) for a in arrays], axis=0
                )
            else:
                combined[key] = np.concatenate(arrays, axis=0)
        step_cache[step] = combined

        v2 = combined["v_l2"].astype(np.float64)
        v3 = combined["v_l3"].astype(np.float64)
        v4 = combined["v_l4"].astype(np.float64)
        weights = combined["weights"].astype(np.float64)
        clipped_z = combined["clipped_z"].astype(np.float64)
        advantages = combined["advantages"].reshape(-1).astype(np.float64)
        valid = combined["loss_mask"].reshape(-1).astype(bool)
        w_valid = weights[valid]
        adv_valid = advantages[valid]
        pos = adv_valid > 0
        neg = adv_valid < 0
        zero = adv_valid == 0

        log_v2 = np.log(v2 + EPS)
        log_v3 = np.log(v3 + EPS)
        log_v4 = np.log(v4 + EPS)
        current_log_by_step[step] = log_v3.reshape(-1)

        csv_consistent = True
        for field in global_csv_fields:
            vals = [float(row[field]) for row in csv_rows]
            csv_consistent &= bool(np.allclose(vals, vals[0], rtol=0, atol=1e-12))
        c = csv_rows[0]

        formula_error = float(
            np.max(np.abs(weights - (1.0 + 0.1 * clipped_z)))
        )
        current_mean_calc = float(log_v3.mean())
        current_std_calc = float(log_v3.std(ddof=0))

        row = {
            "runner_step_internal": step,
            "global_step": step + 1,
            "rank_count": len(rank_dirs),
            "paired_npz_complete": True,
            "rank_global_csv_consistent": csv_consistent,
            "warmup": int(c["warmup"]),
            "history_runner_steps_internal": (
                ";".join(str(x) for x in range(max(0, step - 5), step))
                if step > 0
                else ""
            ),
            "history_steps": int(c["history_steps"]),
            "history_count": int(c["history_count"]),
            "history_log_v_l3_mean": float(c["history_mean"]),
            "history_log_v_l3_std": float(c["history_std"]),
            "current_count": int(c["current_count"]),
            "current_log_v_l3_mean": current_mean_calc,
            "current_log_v_l3_std": current_std_calc,
            "current_csv_mean_abs_error": abs(current_mean_calc - float(c["current_mean"])),
            "current_csv_std_abs_error": abs(current_std_calc - float(c["current_std"])),
            "current_minus_history_log_v_l3": (
                current_mean_calc - float(c["history_mean"])
                if int(c["history_steps"]) > 0
                else np.nan
            ),
            "current_over_history_geomean_ratio_l3": (
                math.exp(current_mean_calc - float(c["history_mean"]))
                if int(c["history_steps"]) > 0
                else np.nan
            ),
            "query_count_all": int(weights.shape[0]),
            "valid_query_count": int(valid.sum()),
            "positive_adv_query_count": int(pos.sum()),
            "negative_adv_query_count": int(neg.sum()),
            "zero_adv_query_count": int(zero.sum()),
            "weight_formula_max_abs_error": formula_error,
            "weight_min_valid": float(w_valid.min()),
            "weight_p05_valid": q(w_valid, 0.05),
            "weight_mean_valid": float(w_valid.mean()),
            "weight_p50_valid": q(w_valid, 0.50),
            "weight_p95_valid": q(w_valid, 0.95),
            "weight_max_valid": float(w_valid.max()),
            "weight_below_one_fraction_valid": float((w_valid < 1.0).mean()),
            "weight_above_one_fraction_valid": float((w_valid > 1.0).mean()),
            "weight_low_clip_fraction_valid": float(
                np.isclose(w_valid, CLIP_LOW, rtol=0, atol=2e-7).mean()
            ),
            "weight_high_clip_fraction_valid": float(
                np.isclose(w_valid, CLIP_HIGH, rtol=0, atol=2e-7).mean()
            ),
            "positive_adv_weight_mean_valid": (
                float(w_valid[pos].mean()) if pos.any() else np.nan
            ),
            "negative_adv_weight_mean_valid": (
                float(w_valid[neg].mean()) if neg.any() else np.nan
            ),
            "negative_minus_positive_adv_weight": (
                float(w_valid[neg].mean() - w_valid[pos].mean())
                if pos.any() and neg.any()
                else np.nan
            ),
            "weight_mean_all": float(weights.mean()),
            "weight_front_h00_24_mean_all": float(weights[:, :25].mean()),
            "weight_back_h25_49_mean_all": float(weights[:, 25:].mean()),
            "weight_back_minus_front_all": float(
                weights[:, 25:].mean() - weights[:, :25].mean()
            ),
            "weight_front_h00_24_mean_valid": float(w_valid[:, :25].mean()),
            "weight_back_h25_49_mean_valid": float(w_valid[:, 25:].mean()),
            "weight_back_minus_front_valid": float(
                w_valid[:, 25:].mean() - w_valid[:, :25].mean()
            ),
            "v_l2_mean": float(v2.mean()),
            "v_l2_median": q(v2, 0.50),
            "v_l2_p05": q(v2, 0.05),
            "v_l2_p95": q(v2, 0.95),
            "v_l3_mean": float(v3.mean()),
            "v_l3_median": q(v3, 0.50),
            "v_l3_p05": q(v3, 0.05),
            "v_l3_p95": q(v3, 0.95),
            "v_l4_mean": float(v4.mean()),
            "v_l4_median": q(v4, 0.50),
            "v_l4_p05": q(v4, 0.05),
            "v_l4_p95": q(v4, 0.95),
            "v_l3_back_over_front_mean": float(
                v3[:, 25:].mean() / v3[:, :25].mean()
            ),
            "h_spearman_with_median_v_l3": spearman_h(np.median(v3, axis=0)),
            "actor_grad_norm": float(c["actor_grad_norm"]),
            "actor_clip_fraction": float(c["actor_clip_fraction"]),
            "actor_approx_kl": float(c["actor_approx_kl"]),
            "actor_ratio": float(c["actor_ratio"]),
            "actor_policy_loss": float(c["actor_policy_loss"]),
            "all_arrays_finite": bool(
                all(
                    np.isfinite(a).all()
                    for a in (v2, v3, v4, weights, clipped_z, advantages)
                )
            ),
        }
        step_rows.append(row)

        for h in range(weights.shape[1]):
            wh = weights[:, h]
            wh_valid = wh[valid]
            h_rows.append(
                {
                    "runner_step_internal": step,
                    "global_step": step + 1,
                    "h": h,
                    "warmup": int(c["warmup"]),
                    "query_count_all": int(weights.shape[0]),
                    "valid_query_count": int(valid.sum()),
                    "v_l2_mean": float(v2[:, h].mean()),
                    "v_l2_median": float(np.median(v2[:, h])),
                    "log_v_l2_mean": float(log_v2[:, h].mean()),
                    "v_l3_mean": float(v3[:, h].mean()),
                    "v_l3_median": float(np.median(v3[:, h])),
                    "log_v_l3_mean": float(log_v3[:, h].mean()),
                    "v_l4_mean": float(v4[:, h].mean()),
                    "v_l4_median": float(np.median(v4[:, h])),
                    "log_v_l4_mean": float(log_v4[:, h].mean()),
                    "weight_mean_all": float(wh.mean()),
                    "weight_mean_valid": float(wh_valid.mean()),
                    "weight_p05_valid": q(wh_valid, 0.05),
                    "weight_p50_valid": q(wh_valid, 0.50),
                    "weight_p95_valid": q(wh_valid, 0.95),
                    "weight_low_clip_fraction_valid": float(
                        np.isclose(wh_valid, CLIP_LOW, rtol=0, atol=2e-7).mean()
                    ),
                    "weight_high_clip_fraction_valid": float(
                        np.isclose(wh_valid, CLIP_HIGH, rtol=0, atol=2e-7).mean()
                    ),
                    "positive_adv_weight_mean_valid": (
                        float(wh_valid[pos].mean()) if pos.any() else np.nan
                    ),
                    "negative_adv_weight_mean_valid": (
                        float(wh_valid[neg].mean()) if neg.any() else np.nan
                    ),
                }
            )

    # Add the post-step rolling window reconstructed from the NPZ values. This is
    # distinct from the pre-step history used to generate that step's weights.
    for row in step_rows:
        step = int(row["runner_step_internal"])
        current_values = current_log_by_step[step]
        row["current_log_v_l3_p05"] = q(current_values, 0.05)
        row["current_log_v_l3_p50"] = q(current_values, 0.50)
        row["current_log_v_l3_p95"] = q(current_values, 0.95)
        pre_window = [
            current_log_by_step[s] for s in range(max(0, step - 5), step)
        ]
        if pre_window:
            pre_values = np.concatenate(pre_window)
            row["history_log_v_l3_p05"] = q(pre_values, 0.05)
            row["history_log_v_l3_p50"] = q(pre_values, 0.50)
            row["history_log_v_l3_p95"] = q(pre_values, 0.95)
        else:
            row["history_log_v_l3_p05"] = np.nan
            row["history_log_v_l3_p50"] = np.nan
            row["history_log_v_l3_p95"] = np.nan
        window = [current_log_by_step[s] for s in range(max(0, step - 4), step + 1)]
        count, mean, std = pooled_mean_std(window)
        row["post_update_history_steps"] = len(window)
        row["post_update_history_count"] = count
        row["post_update_history_log_v_l3_mean"] = mean
        row["post_update_history_log_v_l3_std"] = std
        post_values = np.concatenate(window)
        row["post_update_history_log_v_l3_p05"] = q(post_values, 0.05)
        row["post_update_history_log_v_l3_p50"] = q(post_values, 0.50)
        row["post_update_history_log_v_l3_p95"] = q(post_values, 0.95)

    step_df = pd.DataFrame(step_rows)
    h_df = pd.DataFrame(h_rows)
    step_df.to_csv(HERE / "DVAC_STEP_SUMMARY.csv", index=False, float_format="%.10g")
    h_df.to_csv(HERE / "DVAC_BY_H.csv", index=False, float_format="%.10g")

    control_df = analyze_control_trace(step_cache[0])
    control_df.to_csv(
        HERE / "CONTROL_TRACE_RESET57_DVAC.csv", index=False, float_format="%.10g"
    )

    draw_main_figure(step_df, h_df, control_df)
    write_summary_json(step_df, h_df, control_df)

    return step_df, h_df, control_df


def analyze_control_trace(step0: dict[str, np.ndarray]) -> pd.DataFrame:
    meta_mask = (
        (step0["dvac_meta_source_env_rank"].reshape(-1) == 0)
        & (step0["dvac_meta_local_env_slot"].reshape(-1) == 0)
        & (step0["dvac_meta_reset_id"].reshape(-1) == 57)
    )
    indices = np.flatnonzero(meta_mask)
    if indices.size != 4:
        raise RuntimeError(f"Expected 4 reset57 control-trace queries, found {indices.size}")

    v3 = step0["v_l3"].astype(np.float64)
    weights = step0["weights"].astype(np.float64)
    advantages = step0["advantages"].reshape(-1).astype(np.float64)
    rewards = step0["rewards"].astype(np.float64)
    valid = step0["loss_mask"].reshape(-1).astype(bool)

    rows = []
    for idx in indices:
        query_idx = int(step0["dvac_meta_query_idx"].reshape(-1)[idx])
        values = v3[idx]
        total = float(values.sum())
        peak_h = int(np.argmax(values))
        for h, value in enumerate(values):
            rows.append(
                {
                    "runner_step_internal": 0,
                    "global_step": 1,
                    "source_env_rank": 0,
                    "local_env_slot": 0,
                    "reset_id": 57,
                    "rollout_epoch": int(
                        step0["dvac_meta_rollout_epoch"].reshape(-1)[idx]
                    ),
                    "episode_index_within_step": int(
                        step0["dvac_meta_episode_index_within_step"].reshape(-1)[idx]
                    ),
                    "query_idx": query_idx,
                    "action_slot_start": int(
                        step0["dvac_meta_action_slot_start"].reshape(-1)[idx]
                    ),
                    "h": h,
                    "v_l3": float(value),
                    "v_l3_total_over_h": total,
                    "v_l3_mean_over_h": float(values.mean()),
                    "v_l3_peak_h": peak_h,
                    "v_l3_peak_value": float(values[peak_h]),
                    "weight": float(weights[idx, h]),
                    "advantage": float(advantages[idx]),
                    "reward_sum_over_h": float(rewards[idx].sum()),
                    "loss_mask": bool(valid[idx]),
                    "success_before": bool(
                        step0["dvac_meta_success_before"].reshape(-1)[idx]
                    ),
                }
            )
    frame = pd.DataFrame(rows).sort_values(["query_idx", "h"]).reset_index(drop=True)
    if frame["query_idx"].drop_duplicates().tolist() != [0, 1, 2, 3]:
        raise RuntimeError("Reset57 query indices are not q0..q3")
    return frame


def write_summary_json(step_df: pd.DataFrame, h_df: pd.DataFrame, control_df: pd.DataFrame):
    apply_df = step_df[step_df["warmup"] == 0]
    latest = step_df.iloc[-1]
    apply_h = h_df[h_df["warmup"] == 0]
    avg_h = apply_h.groupby("h", as_index=False).agg(
        weight_mean_all=("weight_mean_all", "mean"),
        weight_mean_valid=("weight_mean_valid", "mean"),
        v_l3_median=("v_l3_median", "median"),
    )
    control_q = (
        control_df.groupby("query_idx", as_index=False)
        .agg(
            action_slot_start=("action_slot_start", "first"),
            v_l3_total_over_h=("v_l3_total_over_h", "first"),
            v_l3_mean_over_h=("v_l3_mean_over_h", "first"),
            v_l3_peak_h=("v_l3_peak_h", "first"),
            v_l3_peak_value=("v_l3_peak_value", "first"),
            advantage=("advantage", "first"),
            loss_mask=("loss_mask", "first"),
        )
        .to_dict(orient="records")
    )
    summary = {
        "snapshot_latest": {
            "runner_step_internal": int(latest["runner_step_internal"]),
            "global_step_completed": int(latest["global_step"]),
            "semantic_note": (
                f"runner_step is zero-based: rollout_step{int(latest['runner_step_internal']):04d} "
                f"and CSV runner_step={int(latest['runner_step_internal'])} correspond to "
                f"the completed UI metric table Global Step {int(latest['global_step'])}/100."
            ),
            "history_used_for_weights_internal_steps": latest[
                "history_runner_steps_internal"
            ],
            "post_update_rolling_window_internal_steps": ";".join(
                str(step)
                for step in range(
                    max(0, int(latest["runner_step_internal"]) - 4),
                    int(latest["runner_step_internal"]) + 1,
                )
            ),
        },
        "latest_metrics": {
            key: (bool(value) if isinstance(value, (bool, np.bool_)) else float(value))
            for key, value in latest.items()
            if key
            in {
                "current_log_v_l3_mean",
                "current_log_v_l3_std",
                "current_log_v_l3_p05",
                "current_log_v_l3_p50",
                "current_log_v_l3_p95",
                "history_log_v_l3_mean",
                "history_log_v_l3_std",
                "history_log_v_l3_p05",
                "history_log_v_l3_p50",
                "history_log_v_l3_p95",
                "current_over_history_geomean_ratio_l3",
                "valid_query_count",
                "weight_mean_valid",
                "weight_p05_valid",
                "weight_p50_valid",
                "weight_p95_valid",
                "weight_low_clip_fraction_valid",
                "weight_high_clip_fraction_valid",
                "positive_adv_weight_mean_valid",
                "negative_adv_weight_mean_valid",
                "negative_minus_positive_adv_weight",
                "weight_back_minus_front_valid",
                "v_l2_median",
                "v_l3_median",
                "v_l4_median",
                "v_l3_back_over_front_mean",
                "h_spearman_with_median_v_l3",
            }
        },
        "apply_steps_aggregate": {
            "global_steps": f"{int(apply_df.global_step.min())}-{int(apply_df.global_step.max())}",
            "weight_mean_valid_weighted": float(
                np.average(apply_df.weight_mean_valid, weights=apply_df.valid_query_count)
            ),
            "weight_low_clip_fraction_valid_weighted": float(
                np.average(
                    apply_df.weight_low_clip_fraction_valid,
                    weights=apply_df.valid_query_count,
                )
            ),
            "weight_high_clip_fraction_valid_weighted": float(
                np.average(
                    apply_df.weight_high_clip_fraction_valid,
                    weights=apply_df.valid_query_count,
                )
            ),
            "negative_minus_positive_adv_weight_mean": float(
                apply_df.negative_minus_positive_adv_weight.mean()
            ),
            "negative_weight_higher_step_count": int(
                (apply_df.negative_minus_positive_adv_weight > 0).sum()
            ),
            "apply_step_count": int(len(apply_df)),
            "back_minus_front_weight_all_mean": float(
                apply_df.weight_back_minus_front_all.mean()
            ),
            "back_minus_front_weight_valid_mean": float(
                apply_df.weight_back_minus_front_valid.mean()
            ),
            "h_spearman_median_v_l3_mean": float(
                apply_df.h_spearman_with_median_v_l3.mean()
            ),
            "h_spearman_apply_aggregate_v_l3": spearman_h(
                avg_h.v_l3_median.to_numpy()
            ),
        },
        "control_trace_reset57": {
            "runner_step_internal": 0,
            "global_step": 1,
            "frames": 200,
            "queries": 4,
            "finish_reason": "step_limit",
            "success": False,
            "query_metrics": control_q,
            "interpretation_boundary": "one failed episode; descriptive alignment only",
        },
    }
    with (HERE / "DVAC_ANALYSIS_SUMMARY.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def get_font(size: int, bold: bool = False):
    names = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
    ]
    for name in names:
        if name.exists():
            return ImageFont.truetype(str(name), size=size)
    return ImageFont.load_default()


FONT_TITLE = get_font(38, bold=True)
FONT_PANEL = get_font(25, bold=True)
FONT_AXIS = get_font(20)
FONT_SMALL = get_font(17)
FONT_TINY = get_font(15)
FONT_BOLD_SMALL = get_font(17, bold=True)

BG = "#f8fafc"
PANEL_BG = "#ffffff"
FG = "#172033"
MUTED = "#586174"
GRID = "#d9dee8"
BLUE = "#2563eb"
ORANGE = "#d97706"
GREEN = "#16835d"
RED = "#cf3c4f"
PURPLE = "#7c3aed"
CYAN = "#0891b2"
GREY = "#7b8497"


def text_size(draw: ImageDraw.ImageDraw, text: str, font):
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def draw_panel(draw, box, title):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=18, fill=PANEL_BG, outline=GRID, width=2)
    draw.text((x0 + 24, y0 + 18), title, fill=FG, font=FONT_PANEL)
    return (x0 + 78, y0 + 78, x1 - 30, y1 - 65)


def map_points(xs, ys, plot, xlim, ylim):
    x0, y0, x1, y1 = plot
    xmin, xmax = xlim
    ymin, ymax = ylim
    px = x0 + (np.asarray(xs) - xmin) / (xmax - xmin) * (x1 - x0)
    py = y1 - (np.asarray(ys) - ymin) / (ymax - ymin) * (y1 - y0)
    return list(zip(px.tolist(), py.tolist()))


def draw_axes(draw, plot, xlim, ylim, xticks, yticks, xlabel="", ylabel="", yfmt=None):
    x0, y0, x1, y1 = plot
    for value in yticks:
        y = map_points([xlim[0]], [value], plot, xlim, ylim)[0][1]
        draw.line((x0, y, x1, y), fill=GRID, width=1)
        label = yfmt(value) if yfmt else f"{value:g}"
        w, h = text_size(draw, label, FONT_TINY)
        draw.text((x0 - w - 10, y - h / 2), label, fill=MUTED, font=FONT_TINY)
    for value in xticks:
        x = map_points([value], [ylim[0]], plot, xlim, ylim)[0][0]
        draw.line((x, y1, x, y1 + 6), fill=MUTED, width=1)
        label = f"{value:g}"
        w, _ = text_size(draw, label, FONT_TINY)
        draw.text((x - w / 2, y1 + 10), label, fill=MUTED, font=FONT_TINY)
    draw.line((x0, y0, x0, y1), fill=MUTED, width=2)
    draw.line((x0, y1, x1, y1), fill=MUTED, width=2)
    if xlabel:
        w, _ = text_size(draw, xlabel, FONT_SMALL)
        draw.text(((x0 + x1 - w) / 2, y1 + 36), xlabel, fill=MUTED, font=FONT_SMALL)
    if ylabel:
        draw.text((x0, y0 - 23), ylabel, fill=MUTED, font=FONT_TINY)


def line(draw, xs, ys, plot, xlim, ylim, color, width=4):
    points = map_points(xs, ys, plot, xlim, ylim)
    if len(points) >= 2:
        draw.line(points, fill=color, width=width, joint="curve")
    for x, y in points[-1:]:
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)


def legend(draw, x, y, items):
    cursor = x
    for label, color in items:
        draw.line((cursor, y + 9, cursor + 26, y + 9), fill=color, width=5)
        cursor += 34
        draw.text((cursor, y), label, fill=FG, font=FONT_TINY)
        w, _ = text_size(draw, label, FONT_TINY)
        cursor += w + 26


def draw_main_figure(step_df, h_df, control_df):
    latest_step = int(step_df.global_step.max())
    x_ticks = sorted(set([1, 10, 20, 30, latest_step]))
    width, height = 2100, 1830
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)
    draw.text((50, 28), "DVAC formal training snapshot — method diagnostics", fill=FG, font=FONT_TITLE)
    draw.text(
        (50, 78),
        f"Paired actor ranks · runner_step 0–{latest_step - 1} = completed Global Steps 1–{latest_step} · snapshot evidence only",
        fill=MUTED,
        font=FONT_AXIS,
    )

    margin, gap = 42, 28
    top = 126
    panel_w = (width - 2 * margin - gap) // 2
    panel_h = 530
    boxes = []
    for row in range(3):
        for col in range(2):
            x0 = margin + col * (panel_w + gap)
            y0 = top + row * (panel_h + gap)
            boxes.append((x0, y0, x0 + panel_w, y0 + panel_h))

    # A. pre-step history and current distribution of the selected signal.
    plot = draw_panel(draw, boxes[0], "A  Selected signal distribution across completed steps")
    x = step_df.global_step.to_numpy()
    current = step_df.current_log_v_l3_mean.to_numpy()
    history = step_df.history_log_v_l3_mean.to_numpy()
    hmask = step_df.history_steps.to_numpy() > 0
    ymin = min(current.min(), history[hmask].min()) - 0.10
    ymax = max(current.max(), history[hmask].max()) + 0.10
    draw_axes(draw, plot, (1, latest_step), (ymin, ymax), x_ticks, np.linspace(ymin, ymax, 5), "Global step", "mean log(V_L3)")
    line(draw, x, current, plot, (1, latest_step), (ymin, ymax), BLUE)
    line(draw, x[hmask], history[hmask], plot, (1, latest_step), (ymin, ymax), ORANGE)
    legend(draw, plot[0] + 10, plot[1] + 8, [("current step", BLUE), ("pre-step recent-5", ORANGE)])
    draw.text((plot[0] + 10, plot[3] - 30), f"latest current {current[-1]:.3f} vs history {history[-1]:.3f}  ({math.exp(current[-1]-history[-1])-1:+.1%} geometric mean)", fill=FG, font=FONT_SMALL)

    # B. L2/L3/L4 medians on a shared log10 scale.
    plot = draw_panel(draw, boxes[1], "B  V_L2 / V_L3 / V_L4 level")
    series = [
        ("V_L2 median", np.log10(step_df.v_l2_median.to_numpy()), BLUE),
        ("V_L3 median", np.log10(step_df.v_l3_median.to_numpy()), GREEN),
        ("V_L4 median", np.log10(step_df.v_l4_median.to_numpy()), PURPLE),
    ]
    all_y = np.concatenate([s[1] for s in series])
    ymin = math.floor(all_y.min() * 2) / 2 - 0.1
    ymax = math.ceil(all_y.max() * 2) / 2 + 0.1
    yticks = np.arange(math.ceil(ymin), math.floor(ymax) + 1)
    draw_axes(draw, plot, (1, latest_step), (ymin, ymax), x_ticks, yticks, "Global step", "median V (log10)", lambda v: f"1e{int(v)}")
    for _, values, color in series:
        line(draw, x, values, plot, (1, latest_step), (ymin, ymax), color)
    legend(draw, plot[0] + 10, plot[1] + 8, [(s[0], s[2]) for s in series])
    draw.text((plot[0] + 10, plot[3] - 30), f"latest medians  L2 {step_df.v_l2_median.iloc[-1]:.4g} · L3 {step_df.v_l3_median.iloc[-1]:.4g} · L4 {step_df.v_l4_median.iloc[-1]:.4g}", fill=FG, font=FONT_SMALL)

    # C. valid-query weights.
    plot = draw_panel(draw, boxes[2], "C  Gradient weights on loss-mask-valid queries")
    p05 = step_df.weight_p05_valid.to_numpy()
    p50 = step_df.weight_p50_valid.to_numpy()
    p95 = step_df.weight_p95_valid.to_numpy()
    mean_w = step_df.weight_mean_valid.to_numpy()
    xlim, ylim = (1, latest_step), (0.78, 1.22)
    draw_axes(draw, plot, xlim, ylim, x_ticks, [0.8, 0.9, 1.0, 1.1, 1.2], "Global step", "weight")
    polygon = map_points(x, p05, plot, xlim, ylim) + list(reversed(map_points(x, p95, plot, xlim, ylim)))
    draw.polygon(polygon, fill="#dbeafe")
    line(draw, x, p50, plot, xlim, ylim, BLUE, width=3)
    line(draw, x, mean_w, plot, xlim, ylim, GREEN, width=4)
    y1 = map_points([1], [1.0], plot, xlim, ylim)[0][1]
    draw.line((plot[0], y1, plot[2], y1), fill=GREY, width=2)
    legend(draw, plot[0] + 10, plot[1] + 8, [("p05–p95", "#93c5fd"), ("median", BLUE), ("mean", GREEN)])
    draw.text((plot[0] + 10, plot[3] - 30), f"latest mean {mean_w[-1]:.3f} · p05 {p05[-1]:.3f} · p95 {p95[-1]:.3f}", fill=FG, font=FONT_SMALL)

    # D. clipping and advantage-sign split.
    plot = draw_panel(draw, boxes[3], "D  Boundary clipping and advantage-sign split")
    apply = step_df[step_df.warmup == 0]
    xa = apply.global_step.to_numpy()
    low = 100 * apply.weight_low_clip_fraction_valid.to_numpy()
    high = 100 * apply.weight_high_clip_fraction_valid.to_numpy()
    gap_adv = 100 * apply.negative_minus_positive_adv_weight.to_numpy()
    max_y = max(10.0, math.ceil(max(high.max(), low.max(), gap_adv.max()) / 2) * 2 + 1)
    draw_axes(draw, plot, (2, latest_step), (0, max_y), sorted(set([2] + x_ticks[1:])), np.linspace(0, max_y, 5), "Global step", "% clipped / weight points", lambda v: f"{v:.0f}")
    line(draw, xa, high, plot, (2, latest_step), (0, max_y), RED)
    line(draw, xa, low, plot, (2, latest_step), (0, max_y), CYAN)
    line(draw, xa, gap_adv, plot, (2, latest_step), (0, max_y), PURPLE)
    legend(draw, plot[0] + 10, plot[1] + 8, [("at 1.2", RED), ("at 0.8", CYAN), ("negative − positive ×100", PURPLE)])
    draw.text((plot[0] + 10, plot[3] - 30), f"latest high {high[-1]:.2f}% · low {low[-1]:.2f}% · neg−pos {gap_adv[-1]:+.2f} points", fill=FG, font=FONT_SMALL)

    # E. future-h effect, both whole trajectory pool and actual valid queries.
    plot = draw_panel(draw, boxes[4], "E  Future-h effect retained by the first method version")
    apply_h = h_df[h_df.warmup == 0]
    avg_h = apply_h.groupby("h", as_index=False).agg(
        weight_all=("weight_mean_all", "mean"),
        weight_valid=("weight_mean_valid", "mean"),
    )
    latest_h = h_df[h_df.runner_step_internal == step_df.runner_step_internal.iloc[-1]]
    hs = avg_h.h.to_numpy()
    values = np.concatenate([avg_h.weight_all, avg_h.weight_valid, latest_h.weight_mean_valid])
    ymin = min(0.98, float(values.min()) - 0.008)
    ymax = max(1.06, float(values.max()) + 0.008)
    draw_axes(draw, plot, (0, 49), (ymin, ymax), [0, 10, 20, 30, 40, 49], np.linspace(ymin, ymax, 5), "future action h", "mean weight", lambda v: f"{v:.3f}")
    line(draw, hs, avg_h.weight_all.to_numpy(), plot, (0, 49), (ymin, ymax), BLUE, width=3)
    line(draw, hs, avg_h.weight_valid.to_numpy(), plot, (0, 49), (ymin, ymax), GREEN, width=4)
    line(draw, hs, latest_h.weight_mean_valid.to_numpy(), plot, (0, 49), (ymin, ymax), ORANGE, width=3)
    legend(draw, plot[0] + 10, plot[1] + 8, [(f"all queries, steps 2–{latest_step}", BLUE), (f"valid, steps 2–{latest_step}", GREEN), ("valid, latest", ORANGE)])
    mean_delta = step_df.loc[step_df.warmup == 0, "weight_back_minus_front_valid"].mean()
    draw.text((plot[0] + 10, plot[3] - 30), f"apply-step mean: h25–49 minus h0–24 = {mean_delta:+.4f}", fill=FG, font=FONT_SMALL)

    # F. the single recorded failed episode, q0..q3 by h.
    plot = draw_panel(draw, boxes[5], "F  Recorded reset57 failure: q0–q3 V_L3(h), step 1 warmup")
    matrix = control_df.pivot(index="query_idx", columns="h", values="v_l3").to_numpy()
    log_matrix = np.log10(matrix + EPS)
    x0, y0, x1, y1 = plot
    heat_top = y0 + 46
    heat_bottom = y1 - 54
    cell_w = (x1 - x0) / 50
    cell_h = (heat_bottom - heat_top) / 4
    lo, hi = float(np.quantile(log_matrix, 0.02)), float(np.quantile(log_matrix, 0.98))
    for qi in range(4):
        for h in range(50):
            t = float(np.clip((log_matrix[qi, h] - lo) / (hi - lo), 0, 1))
            # Perceptually ordered blue -> teal -> amber ramp.
            if t < 0.5:
                u = t / 0.5
                c0, c1 = (38, 78, 153), (30, 160, 160)
            else:
                u = (t - 0.5) / 0.5
                c0, c1 = (30, 160, 160), (239, 165, 48)
            color = tuple(int(a + u * (b - a)) for a, b in zip(c0, c1))
            rx0 = x0 + h * cell_w
            ry0 = heat_top + qi * cell_h
            draw.rectangle((rx0, ry0, rx0 + cell_w + 1, ry0 + cell_h + 1), fill=color)
        draw.text((x0 - 44, heat_top + qi * cell_h + cell_h / 2 - 10), f"q{qi}", fill=FG, font=FONT_SMALL)
    for tick in [0, 10, 20, 30, 40, 49]:
        tx = x0 + (tick + 0.5) * cell_w
        label = str(tick)
        tw, _ = text_size(draw, label, FONT_TINY)
        draw.text((tx - tw / 2, heat_bottom + 9), label, fill=MUTED, font=FONT_TINY)
    draw.text((x0, y0 + 7), "lower", fill=MUTED, font=FONT_TINY)
    grad_x = x0 + 58
    for i in range(120):
        t = i / 119
        if t < 0.5:
            u = t / 0.5
            c0, c1 = (38, 78, 153), (30, 160, 160)
        else:
            u = (t - 0.5) / 0.5
            c0, c1 = (30, 160, 160), (239, 165, 48)
        color = tuple(int(a + u * (b - a)) for a, b in zip(c0, c1))
        draw.line((grad_x + i, y0 + 8, grad_x + i, y0 + 24), fill=color, width=1)
    draw.text((grad_x + 128, y0 + 7), "higher log10 V_L3", fill=MUTED, font=FONT_TINY)
    qsummary = control_df.groupby("query_idx").first()
    totals = " · ".join(f"q{qi} total {qsummary.loc[qi, 'v_l3_total_over_h']:.3f}" for qi in range(4))
    draw.text((x0, y1 - 27), totals, fill=FG, font=FONT_SMALL)

    footer = (
        "Definitions: current/history use all 1,024 trajectory queries × 50 h per step; weight, clipping, and advantage split use loss-mask-valid queries. "
        "Reset57 is one failed episode and is shown only for alignment."
    )
    draw.text((50, height - 44), footer, fill=MUTED, font=FONT_SMALL)
    img.save(HERE / f"DVAC_FORMAL_STEP{latest_step}_METHOD_OVERVIEW.png", optimize=True)


if __name__ == "__main__":
    step_df, h_df, control_df = analyze()
    print(f"wrote {len(step_df)} step rows, {len(h_df)} h rows, {len(control_df)} control rows")
    print(step_df.tail(3).to_string(index=False))
