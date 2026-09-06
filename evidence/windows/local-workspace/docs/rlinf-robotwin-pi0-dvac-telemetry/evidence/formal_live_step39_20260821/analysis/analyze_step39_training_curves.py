from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE.parent
WORKSPACE = Path(__file__).resolve().parents[5]
OLD_ANALYSIS = (
    WORKSPACE
    / "docs"
    / "rlinf-robotwin-pi0-dvac-telemetry"
    / "evidence"
    / "formal_live_step22_20260821"
    / "analysis"
)
BASELINE_JSON = WORKSPACE / "audits" / "20260717-084926-grpo-current" / "analysis.json"


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def regenerate_base_tables() -> None:
    train = import_file("step39_build_training", OLD_ANALYSIS / "build_training_analysis.py")
    train.HERE = HERE
    train.SNAPSHOT = SNAPSHOT
    train.METRICS_LOG = SNAPSHOT / "run" / "metrics.log"
    train.BASELINE_JSON = BASELINE_JSON
    launch_time = SNAPSHOT / "runtime" / "launch_started_at.txt"
    if not launch_time.exists():
        launch_time = OLD_ANALYSIS.parent / "runtime" / "launch_started_at.txt"
    train.LAUNCH_TIME_FILE = launch_time
    train.main()

    dvac = import_file("step39_analyze_dvac", HERE / "analyze_dvac_snapshot.py")
    dvac.HERE = HERE
    dvac.SNAPSHOT = SNAPSHOT
    dvac.DVAC_DIR = SNAPSHOT / "run" / "dvac_train"
    dvac.CONTROL_DIR = (
        SNAPSHOT
        / "run"
        / "control_trace"
        / "adjust_bottle"
        / "worker_000"
        / "env_slot_000"
        / "recording_0000_episode_0000_reset_57"
    )
    dvac.analyze()


def rolling(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def safe_mean(series: pd.Series) -> float:
    return float(series.dropna().mean())


def success_window(
    current: pd.DataFrame, baseline: pd.DataFrame, start: int, end: int
) -> dict[str, float | int]:
    cur = current[current.global_step.between(start, end)]
    base = baseline[baseline.step.between(start, end)]
    n_cur = int(len(cur) * 256)
    n_base = int(len(base) * 256)
    p_cur = float(cur.success_once.mean())
    p_base = float(base.success_once.mean())
    # Descriptive binomial scale only. Sequential on-policy steps are not an
    # independent hypothesis test, but this gives the right order of magnitude.
    se = math.sqrt(
        p_cur * (1 - p_cur) / n_cur + p_base * (1 - p_base) / n_base
    )
    return {
        "start": start,
        "end": end,
        "steps": len(cur),
        "current_mean": p_cur,
        "baseline_mean": p_base,
        "delta": p_cur - p_base,
        "descriptive_independent_binomial_se": se,
        "delta_over_descriptive_se": (p_cur - p_base) / se,
    }


def build_comparison() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    current = pd.read_csv(HERE / "TRAIN_METRICS.csv")
    baseline = pd.DataFrame(json.loads(BASELINE_JSON.read_text(encoding="utf-8"))["metrics"])
    dvac = pd.read_csv(HERE / "DVAC_STEP_SUMMARY.csv")
    by_h = pd.read_csv(HERE / "DVAC_BY_H.csv")

    overlap = int(current.global_step.max())
    baseline_overlap = baseline[baseline.step <= overlap].copy()
    merged = current.merge(
        baseline_overlap,
        left_on="global_step",
        right_on="step",
        suffixes=("_dvac", "_grpo"),
        how="left",
    )
    for window in (5, 10, 20):
        merged[f"success_roll{window}_dvac"] = rolling(
            merged.success_once_dvac, window
        )
        merged[f"success_roll{window}_grpo"] = rolling(
            merged.success_once_grpo, window
        )
        merged[f"success_roll{window}_delta"] = (
            merged[f"success_roll{window}_dvac"]
            - merged[f"success_roll{window}_grpo"]
        )

    merged["delta_success"] = merged.success_once_dvac - merged.success_once_grpo
    merged["delta_kl"] = merged["actor/approx_kl_dvac"] - merged["actor/approx_kl_grpo"]
    merged["delta_clip_fraction"] = (
        merged["actor/clip_fraction_dvac"] - merged["actor/clip_fraction_grpo"]
    )
    merged["delta_grad_norm"] = (
        merged["actor/grad_norm_dvac"] - merged["actor/grad_norm_grpo"]
    )
    merged["delta_step_time_s"] = merged.step_time_s_dvac - merged.step_time_s_grpo
    merged.to_csv(HERE / "STEP39_EFFECT_COMPARISON.csv", index=False, float_format="%.10g")

    windows = [
        success_window(current, baseline, 1, 1),
        success_window(current, baseline, 1, 10),
        success_window(current, baseline, 11, 20),
        success_window(current, baseline, 21, 30),
        success_window(current, baseline, 31, 39),
        success_window(current, baseline, 1, 39),
        success_window(current, baseline, 20, 39),
        success_window(current, baseline, 30, 39),
        success_window(current, baseline, 35, 39),
    ]

    apply = dvac[dvac.warmup == 0].copy()
    latest = apply.iloc[-1]
    latest_h = by_h[by_h.global_step == overlap]
    apply_h = by_h[by_h.warmup == 0]
    h_avg = apply_h.groupby("h", as_index=False).agg(
        v_l3_median=("v_l3_median", "median"),
        weight_valid=("weight_mean_valid", "mean"),
        weight_all=("weight_mean_all", "mean"),
    )
    latest_h_small = latest_h[
        ["h", "v_l3_median", "weight_mean_valid", "weight_mean_all"]
    ].rename(
        columns={
            "v_l3_median": "latest_v_l3_median",
            "weight_mean_valid": "latest_weight_valid",
            "weight_mean_all": "latest_weight_all",
        }
    )
    h_summary = h_avg.merge(latest_h_small, on="h", how="left")
    h_summary.to_csv(HERE / "STEP39_FUTURE_H_SUMMARY.csv", index=False, float_format="%.10g")

    baseline_phases = []
    for start, end in ((1, 39), (40, 59), (60, 79), (80, 100)):
        block = baseline[baseline.step.between(start, end)]
        baseline_phases.append(
            {
                "start": start,
                "end": end,
                "mean_success": float(block.success_once.mean()),
                "min_success": float(block.success_once.min()),
                "max_success": float(block.success_once.max()),
                "mean_kl": float(block["actor/approx_kl"].mean()),
                "mean_clip_fraction": float(block["actor/clip_fraction"].mean()),
                "mean_grad_norm": float(block["actor/grad_norm"].mean()),
            }
        )

    summary = {
        "snapshot_global_step": overlap,
        "success_windows": windows,
        "latest": {
            "current_success": float(current.iloc[-1].success_once),
            "baseline_success": float(baseline_overlap.iloc[-1].success_once),
            "delta_success": float(
                current.iloc[-1].success_once - baseline_overlap.iloc[-1].success_once
            ),
            "current_kl": float(current.iloc[-1]["actor/approx_kl"]),
            "baseline_kl": float(baseline_overlap.iloc[-1]["actor/approx_kl"]),
            "current_clip_fraction": float(
                current.iloc[-1]["actor/clip_fraction"]
            ),
            "baseline_clip_fraction": float(
                baseline_overlap.iloc[-1]["actor/clip_fraction"]
            ),
            "current_grad_norm": float(current.iloc[-1]["actor/grad_norm"]),
            "baseline_grad_norm": float(
                baseline_overlap.iloc[-1]["actor/grad_norm"]
            ),
            "current_ratio": float(current.iloc[-1]["actor/ratio"]),
            "baseline_ratio": float(baseline_overlap.iloc[-1]["actor/ratio"]),
        },
        "overlap_means": {
            "current_kl": float(current["actor/approx_kl"].mean()),
            "baseline_kl": float(baseline_overlap["actor/approx_kl"].mean()),
            "current_clip_fraction": float(
                current["actor/clip_fraction"].mean()
            ),
            "baseline_clip_fraction": float(
                baseline_overlap["actor/clip_fraction"].mean()
            ),
            "current_grad_norm": float(current["actor/grad_norm"].mean()),
            "baseline_grad_norm": float(
                baseline_overlap["actor/grad_norm"].mean()
            ),
            "current_step_time_s": float(current.step_time_s.mean()),
            "baseline_step_time_s": float(baseline_overlap.step_time_s.mean()),
        },
        "dvac_apply": {
            "steps": int(len(apply)),
            "latest_current_log_v_l3_mean": float(latest.current_log_v_l3_mean),
            "latest_history_log_v_l3_mean": float(latest.history_log_v_l3_mean),
            "latest_current_minus_history": float(
                latest.current_minus_history_log_v_l3
            ),
            "latest_weight_p05": float(latest.weight_p05_valid),
            "latest_weight_median": float(latest.weight_p50_valid),
            "latest_weight_mean": float(latest.weight_mean_valid),
            "latest_weight_p95": float(latest.weight_p95_valid),
            "latest_low_clip_fraction": float(latest.weight_low_clip_fraction_valid),
            "latest_high_clip_fraction": float(
                latest.weight_high_clip_fraction_valid
            ),
            "apply_mean_weight_valid": float(apply.weight_mean_valid.mean()),
            "apply_mean_weight_all": float(apply.weight_mean_all.mean()),
            "apply_mean_p05": float(apply.weight_p05_valid.mean()),
            "apply_mean_p95": float(apply.weight_p95_valid.mean()),
            "apply_mean_high_clip_fraction": float(
                apply.weight_high_clip_fraction_valid.mean()
            ),
            "apply_mean_low_clip_fraction": float(
                apply.weight_low_clip_fraction_valid.mean()
            ),
            "apply_mean_negative_minus_positive_weight": float(
                apply.negative_minus_positive_adv_weight.mean()
            ),
            "apply_mean_back_minus_front_valid": float(
                apply.weight_back_minus_front_valid.mean()
            ),
            "mean_preclip_grad_norm": float(apply.actor_grad_norm.mean()),
            "min_preclip_grad_norm": float(apply.actor_grad_norm.min()),
            "mean_global_clip_rescale_if_norm_1": float(
                (1.0 / apply.actor_grad_norm).mean()
            ),
            "mean_high_to_low_boundary_ratio": 1.2 / 0.8,
            "mean_p95_to_p05_ratio": float(
                (apply.weight_p95_valid / apply.weight_p05_valid).mean()
            ),
        },
        "historical_grpo_phases": baseline_phases,
        "interpretation": {
            "causal_limit": (
                "The historical run is not a paired off arm. Global step 1 already differs "
                "before DVAC weighting is applied, so small later success gaps cannot be "
                "attributed to the method."
            ),
            "success_metric": (
                "success_once is on-policy training-rollout success, not held-out evaluation."
            ),
            "global_clip": (
                "All observed pre-clip gradient norms exceed clip_grad=1. Uniform rescaling "
                "would be canceled; per-h weights matter mainly through relative gradient "
                "composition and direction."
            ),
        },
    }
    (HERE / "STEP39_EFFECT_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return merged, dvac, h_summary, summary


def style_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dbe3ef", linewidth=0.8)
    ax.tick_params(colors="#53627a", labelsize=9)
    ax.title.set_color("#16233b")
    ax.xaxis.label.set_color("#53627a")
    ax.yaxis.label.set_color("#53627a")


def plot_effect(merged: pd.DataFrame, summary: dict) -> None:
    blue, orange, green, purple = "#2563eb", "#ea580c", "#15803d", "#7c3aed"
    x = merged.global_step.to_numpy()
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    fig.patch.set_facecolor("#f8fafc")
    fig.suptitle(
        "DVAC-GRPO vs historical GRPO through Global Step 39",
        fontsize=20,
        fontweight="bold",
        color="#16233b",
    )
    for ax in axes.flat:
        ax.set_facecolor("white")
        style_axes(ax)

    ax = axes[0, 0]
    ax.plot(x, 100 * merged.success_once_dvac, color=orange, alpha=0.25, lw=1)
    ax.plot(x, 100 * merged.success_once_grpo, color=blue, alpha=0.25, lw=1)
    ax.plot(x, 100 * merged.success_roll10_dvac, color=orange, lw=2.5, label="DVAC 10-step mean")
    ax.plot(x, 100 * merged.success_roll10_grpo, color=blue, lw=2.5, label="GRPO 10-step mean")
    ax.set_title("Training-rollout success")
    ax.set_ylabel("success (%)")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 1]
    ax.axhline(0, color="#64748b", lw=1)
    ax.plot(x, 100 * merged.success_roll5_delta, color="#f59e0b", lw=1.5, label="5-step delta")
    ax.plot(x, 100 * merged.success_roll10_delta, color=purple, lw=2.5, label="10-step delta")
    ax.plot(x, 100 * merged.success_roll20_delta, color=green, lw=2, label="20-step delta")
    ax.set_title("Smoothed DVAC minus GRPO success")
    ax.set_ylabel("percentage points")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 0]
    ax.plot(x, merged["actor/approx_kl_dvac"], color=orange, lw=2, label="DVAC KL")
    ax.plot(x, merged["actor/approx_kl_grpo"], color=blue, lw=2, label="GRPO KL")
    ax.set_title("Approx KL: policy movement")
    ax.set_ylabel("approx KL")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 1]
    ax.plot(x, 100 * merged["actor/clip_fraction_dvac"], color=orange, lw=2, label="DVAC")
    ax.plot(x, 100 * merged["actor/clip_fraction_grpo"], color=blue, lw=2, label="GRPO")
    ax.set_title("PPO joint-chunk clip fraction")
    ax.set_ylabel("clipped samples (%)")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[2, 0]
    ax.plot(x, merged["actor/grad_norm_dvac"], color=orange, lw=2, label="DVAC")
    ax.plot(x, merged["actor/grad_norm_grpo"], color=blue, lw=2, label="GRPO")
    ax.axhline(1, color="#64748b", ls="--", lw=1, label="global clip = 1")
    ax.set_title("Pre-clip gradient norm")
    ax.set_ylabel("norm")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[2, 1]
    ax.plot(x, merged.step_time_s_dvac / 60, color=orange, lw=2, label="DVAC")
    ax.plot(x, merged.step_time_s_grpo / 60, color=blue, lw=2, label="GRPO")
    ax.set_title("Wall time per global step")
    ax.set_ylabel("minutes")
    ax.set_xlabel("global step")
    ax.legend(frameon=False, fontsize=9)
    axes[2, 0].set_xlabel("global step")

    window = next(w for w in summary["success_windows"] if w["start"] == 30 and w["end"] == 39)
    fig.text(
        0.5,
        -0.015,
        f"Recent g30–39 success delta: {100*window['delta']:+.2f} pp.  "
        "Step 1 (before DVAC applies) already differed by -6.25 pp; this is a historical, not paired, comparison.",
        ha="center",
        fontsize=10,
        color="#53627a",
    )
    fig.savefig(HERE / "STEP39_TRAINING_EFFECT_VS_GRPO.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_method(dvac: pd.DataFrame, h_summary: pd.DataFrame, summary: dict) -> None:
    blue, orange, green, purple = "#2563eb", "#ea580c", "#15803d", "#7c3aed"
    apply = dvac[dvac.warmup == 0].copy()
    x = dvac.global_step.to_numpy()
    xa = apply.global_step.to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    fig.patch.set_facecolor("#f8fafc")
    fig.suptitle(
        "What the DVAC weighting actually changes",
        fontsize=20,
        fontweight="bold",
        color="#16233b",
    )
    for ax in axes.flat:
        ax.set_facecolor("white")
        style_axes(ax)

    ax = axes[0, 0]
    ax.plot(x, dvac.current_log_v_l3_mean, color=orange, lw=2.5, label="current log V_L3")
    ax.plot(x[1:], dvac.history_log_v_l3_mean.iloc[1:], color=blue, lw=2, label="recent-5 history")
    ax.set_title("Signal level and moving reference")
    ax.set_ylabel("mean natural log V_L3")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 1]
    ax.fill_between(
        xa,
        apply.weight_p05_valid,
        apply.weight_p95_valid,
        color="#bfdbfe",
        alpha=0.7,
        label="p05–p95",
    )
    ax.plot(xa, apply.weight_p50_valid, color=blue, lw=2, label="median")
    ax.plot(xa, apply.weight_mean_valid, color=green, lw=2, label="mean")
    ax.axhline(1, color="#64748b", lw=1)
    ax.set_title("Weights on gradient-contributing queries")
    ax.set_ylabel("per-action weight")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 0]
    ax.plot(
        h_summary.h,
        h_summary.weight_valid,
        color=green,
        lw=2.5,
        label="apply-step mean",
    )
    ax.plot(
        h_summary.h,
        h_summary.latest_weight_valid,
        color=orange,
        lw=2,
        label="latest step",
    )
    ax.axhline(1, color="#64748b", lw=1)
    ax.set_title("Future-h weighting: later actions get more")
    ax.set_xlabel("future action index h")
    ax.set_ylabel("mean valid weight")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 1]
    ax.plot(
        xa,
        100 * apply.weight_high_clip_fraction_valid,
        color=orange,
        lw=2,
        label="at upper bound 1.2",
    )
    ax.plot(
        xa,
        100 * apply.weight_low_clip_fraction_valid,
        color=blue,
        lw=2,
        label="at lower bound 0.8",
    )
    ax.plot(
        xa,
        100 * apply.negative_minus_positive_adv_weight,
        color=purple,
        lw=2,
        label="negative minus positive weight",
    )
    ax.set_title("Asymmetry: upper clipping and negative credit")
    ax.set_xlabel("global step")
    ax.set_ylabel("percentage points / clipped points (%)")
    ax.legend(frameon=False, fontsize=9)

    method = summary["dvac_apply"]
    fig.text(
        0.5,
        -0.02,
        f"Apply mean p05–p95 ratio {method['mean_p95_to_p05_ratio']:.2f}×; hard boundary ratio 1.50×.  "
        f"All pre-clip norms exceed 1 (minimum {method['min_preclip_grad_norm']:.1f}), so global clipping makes relative direction the main effect.",
        ha="center",
        fontsize=10,
        color="#53627a",
    )
    fig.savefig(HERE / "STEP39_DVAC_WEIGHT_MECHANICS.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    regenerate_base_tables()
    merged, dvac, h_summary, summary = build_comparison()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
