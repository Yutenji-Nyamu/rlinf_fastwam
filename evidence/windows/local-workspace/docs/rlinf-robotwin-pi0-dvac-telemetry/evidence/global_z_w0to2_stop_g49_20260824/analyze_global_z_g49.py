from __future__ import annotations

import importlib.util
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
RAW = HERE / "raw"
OUT = HERE / "analysis"
BASE_ANALYSIS = (
    WORKSPACE
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/"
    "r_only_formal_live_g23_20260822/analysis/analyze_r_only_g23.py"
)
PLOT_ANALYSIS = (
    WORKSPACE
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/"
    "r_only_formal_live_g50_20260822/analyze_live_g50.py"
)
GLOBAL_Z_ANALYSIS = (
    WORKSPACE
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/"
    "global_z_w0to2_live_g35_20260824/analyze_global_z_g35.py"
)
THREE_RUN_HISTORY = (
    WORKSPACE
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/"
    "r_only_formal_stop_g51_20260822/analysis/THREE_RUN_METRICS_G51.csv"
)
V3_HISTORY = (
    WORKSPACE
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/"
    "v3_formal_stop_g49_20260823/analysis/FOUR_RUN_TRAINING_G49.csv"
)

CURRENT_LABEL = "DVAC global-z [0,2]"
LATEST_EXPECTED = 49
LATEST_RUNNER_STEP = 48


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_smoothing(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["step"] = pd.to_numeric(result["step"]).astype(int)
    result["success_pct"] = pd.to_numeric(result["success_pct"])
    result = result.drop_duplicates(["run", "step"], keep="last").sort_values(["run", "step"])
    result["success_roll5_pct"] = result.groupby("run", sort=False)["success_pct"].transform(
        lambda series: series.rolling(5, min_periods=1).mean()
    )
    result["success_roll10_pct"] = result.groupby("run", sort=False)["success_pct"].transform(
        lambda series: series.rolling(10, min_periods=1).mean()
    )
    return result.reset_index(drop=True)


def load_full_histories() -> pd.DataFrame:
    """Load every comparator through its real endpoint; never forward-fill."""
    three = pd.read_csv(THREE_RUN_HISTORY)
    keep = ["run", "step", "success_pct", "approx_kl", "clip_pct", "grad_norm", "step_min"]
    three = three[keep].copy()

    v3_all = pd.read_csv(V3_HISTORY)
    v3 = v3_all[v3_all.run == "DVAC v3 R-only [0,2]"][keep].copy()
    historical = pd.concat([three, v3], ignore_index=True, sort=False)
    return historical.drop_duplicates(["run", "step"], keep="last")


def run_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    frame = frame.sort_values("step")
    return {
        "steps": int(frame.step.max()),
        "mean_success_pct": float(frame.success_pct.mean()),
        "latest_success_pct": float(frame.iloc[-1].success_pct),
        "latest5_success_pct": float(frame.iloc[-1].success_roll5_pct),
        "latest10_success_pct": float(frame.iloc[-1].success_roll10_pct),
    }


def plot_current_smoothing(plot, current: pd.DataFrame, latest: int) -> None:
    current = current.sort_values("step")
    x = current.step.to_numpy(float)
    image, draw = plot.make_canvas(
        f"Global-z [0,2] training-rollout success through Global Step {latest}",
        "Raw per-step success plus trailing 5-step and 10-step means; step 50 is incomplete and excluded.",
        760,
    )
    plot.draw_chart(
        draw,
        (55, 165, 1745, 720),
        "Raw and smoothed success",
        [
            ("each step", x, current.success_pct.to_numpy(float), "#94A3B8"),
            ("trailing 5-step mean", x, current.success_roll5_pct.to_numpy(float), plot.ORANGE),
            ("trailing 10-step mean", x, current.success_roll10_pct.to_numpy(float), plot.GREEN),
        ],
        "% trajectories successful",
        x_limits=(1, latest),
    )
    image.save(OUT / "GLOBAL_Z_SUCCESS_RAW_ROLL5_ROLL10_G49.png", optimize=True)


def plot_five_run_smoothing(plot, metrics: pd.DataFrame) -> None:
    colors = {
        "Original GRPO": plot.BLUE,
        "DVAC v1 global-z": plot.GOLD,
        "DVAC v2 R-only": plot.PURPLE,
        "DVAC v3 R-only [0,2]": plot.ORANGE,
        CURRENT_LABEL: plot.GREEN,
    }
    endpoints = {
        label: int(metrics[metrics.run == label].step.max())
        for label in colors
    }
    endpoint_text = ", ".join(
        f"{label.replace('DVAC ', '')}: g{endpoints[label]}" for label in colors
    )
    image, draw = plot.make_canvas(
        "Five training-rollout success histories: raw, 5-step, and 10-step",
        f"Each line stops at its real endpoint; no forward fill. {endpoint_text}.",
        1610,
    )
    boxes = [(55, 175, 1745, 625), (55, 665, 1745, 1115), (55, 1155, 1745, 1570)]
    specs = [
        ("success_pct", "A. Success at each global step"),
        ("success_roll5_pct", "B. Trailing 5-step mean"),
        ("success_roll10_pct", "C. Trailing 10-step mean"),
    ]
    for index, (box, (column, title)) in enumerate(zip(boxes, specs)):
        series = []
        for label, color in colors.items():
            frame = metrics[metrics.run == label].sort_values("step")
            series.append(
                (label, frame.step.to_numpy(float), frame[column].to_numpy(float), color)
            )
        plot.draw_chart(
            draw,
            box,
            title,
            series,
            "% trajectories successful",
            x_limits=(1, int(metrics.step.max())),
            legend=True,
        )
    image.save(OUT / "FIVE_RUN_SUCCESS_RAW_ROLL5_ROLL10_G49.png", optimize=True)


def resource_summary(resources: pd.DataFrame) -> dict[str, float | int | str]:
    resources = resources.copy()
    time = resources.drop_duplicates("elapsed_s", keep="last").sort_values("elapsed_s")
    gpu0 = resources[resources.gpu_index == 0]
    gpu1 = resources[resources.gpu_index == 1]
    return {
        "through_timestamp": str(time.iloc[-1].timestamp),
        "elapsed_hours": float(time.iloc[-1].elapsed_s / 3600),
        "gpu0_peak_gib": float(gpu0.gpu_memory_used_mib.max() / 1024),
        "gpu1_peak_gib": float(gpu1.gpu_memory_used_mib.max() / 1024),
        "cgroup_start_gib": float(time.iloc[0].cgroup_current_bytes / 2**30),
        "cgroup_latest_gib": float(time.iloc[-1].cgroup_current_bytes / 2**30),
        "cgroup_peak_gib": float(time.cgroup_current_bytes.max() / 2**30),
        "event_max_delta": int(time.iloc[-1].event_max - time.iloc[0].event_max),
        "event_oom_delta": int(time.iloc[-1].event_oom - time.iloc[0].event_oom),
        "event_oom_kill_delta": int(time.iloc[-1].event_oom_kill - time.iloc[0].event_oom_kill),
        "host_available_latest_gib": float(time.iloc[-1].host_mem_available_kib / 2**20),
        "disk_available_latest_gib": float(time.iloc[-1].disk_available_kib / 2**20),
    }


def main() -> None:
    base = load_module("global_z_g49_base", BASE_ANALYSIS)
    plot = load_module("global_z_g49_plot", PLOT_ANALYSIS)
    global_z = load_module("global_z_g49_helpers", GLOBAL_Z_ANALYSIS)
    OUT.mkdir(parents=True, exist_ok=True)

    main_metrics = base.parse_metrics(RAW / "run/metrics.log")
    latest = int(main_metrics.global_step.max())
    if latest != LATEST_EXPECTED:
        raise RuntimeError(f"snapshot contract expected g{LATEST_EXPECTED}, parsed g{latest}")

    current = global_z.standardize_current(main_metrics)
    historical = load_full_histories()
    metrics = add_smoothing(pd.concat([historical, current], ignore_index=True, sort=False))
    current = metrics[metrics.run == CURRENT_LABEL].copy()

    plot_current_smoothing(plot, current, latest)
    plot_five_run_smoothing(plot, metrics)
    metrics.to_csv(OUT / "FIVE_RUN_SUCCESS_RAW_ROLL5_ROLL10_G49.csv", index=False)
    current.to_csv(OUT / "GLOBAL_Z_SUCCESS_RAW_ROLL5_ROLL10_G49.csv", index=False)

    rank_paths = sorted((RAW / "run/dvac_train").glob("actor_rank*/runner_step_metrics.csv"))
    rank, dvac = base.aggregate_dvac_steps(rank_paths)
    npz_paths = sorted(
        (RAW / "run/dvac_train").glob(
            f"actor_rank*/rollout_step{LATEST_RUNNER_STEP:04d}.npz"
        )
    )
    _, horizon, method = global_z.load_latest(npz_paths)
    resources = pd.read_csv(RAW / "runtime/resource_monitor/resources.csv")

    # Reuse the established method/resource renderers, but keep this snapshot's names.
    global_z.OUT = OUT
    global_z.plot_method(plot, dvac, horizon, latest, method)
    (OUT / "GLOBAL_Z_METHOD_DIAGNOSTICS_G35.png").replace(
        OUT / "GLOBAL_Z_METHOD_DIAGNOSTICS_G49.png"
    )
    global_z.plot_resources(plot, resources)
    (OUT / "GLOBAL_Z_RESOURCES_G35.png").replace(OUT / "GLOBAL_Z_RESOURCES_G49.png")

    rank.to_csv(OUT / "GLOBAL_Z_DVAC_RANK_METRICS_G49.csv", index=False)
    dvac.to_csv(OUT / "GLOBAL_Z_DVAC_STEP_METRICS_G49.csv", index=False)
    horizon.to_csv(OUT / "GLOBAL_Z_LATEST_HORIZON_G49.csv", index=False)
    resources.to_csv(OUT / "GLOBAL_Z_RESOURCES_G49.csv", index=False)

    summaries = {
        label: run_summary(metrics[metrics.run == label])
        for label in metrics.run.drop_duplicates().tolist()
    }
    current_summary = summaries[CURRENT_LABEL]
    common_horizon = metrics[metrics.step <= latest]
    common_summaries = {
        label: run_summary(common_horizon[common_horizon.run == label])
        for label in common_horizon.run.drop_duplicates().tolist()
    }
    comparisons = {}
    for label, other in common_summaries.items():
        if label == CURRENT_LABEL:
            continue
        comparisons[label] = {
            "mean_success_pp_g1_to_g49": current_summary["mean_success_pct"] - other["mean_success_pct"],
            "latest5_success_pp_at_g49": current_summary["latest5_success_pct"] - other["latest5_success_pct"],
            "latest10_success_pp_at_g49": current_summary["latest10_success_pct"] - other["latest10_success_pct"],
        }

    apply = dvac[dvac.warmup == 0]
    driver_text = (RAW / "runtime/driver.log").read_text(encoding="utf-8", errors="replace")
    fatal_pattern = re.compile(
        r"ActorDiedError|RayActorError|CUDA out of memory|OutOfMemory|NCCL.*(?:error|Error)|fatal|FATAL"
    )
    numeric = main_metrics.select_dtypes(include=[np.number]).to_numpy(float)
    summary = {
        "snapshot": {
            "latest_complete_global_step": latest,
            "incomplete_global_step_excluded": 50,
            "latest_npz_rank_count": len(npz_paths),
            "main_metric_rows": int(len(main_metrics)),
            "all_main_numeric_finite": bool(np.isfinite(numeric).all()),
            "fatal_matches_in_driver_log": len(fatal_pattern.findall(driver_text)),
            "known_optional_curobo_import_tracebacks": driver_text.count("No module named 'curobo"),
        },
        "smoothing": {
            "kind": "trailing arithmetic mean",
            "windows": [1, 5, 10],
            "min_periods": 1,
            "forward_fill": False,
        },
        "run_summaries_full_real_endpoints": summaries,
        "current_minus_comparators_on_common_g1_to_g49": comparisons,
        "global_z_method_latest_g49": method,
        "global_z_method_apply_g2_to_g49": {
            "weight_p05_mean": float(apply.weight_p05.mean()),
            "weight_median_mean": float(apply.weight_p50.mean()),
            "weight_p95_mean": float(apply.weight_p95.mean()),
            "weight_mean": float(apply.weight_mean.mean()),
            "downweighted_fraction_mean": float(apply.weight_below_one_fraction.mean()),
            "upweighted_fraction_mean": float(apply.weight_above_one_fraction.mean()),
            "at_zero_fraction_mean": float(apply.z_low_clip_fraction.mean()),
            "at_two_fraction_mean": float(apply.z_high_clip_fraction.mean()),
            "positive_adv_weight_mean": float(apply.positive_adv_weight_mean.mean()),
            "negative_adv_weight_mean": float(apply.negative_adv_weight_mean.mean()),
            "raw_v_geometric_change_g1_to_g49": float(
                math.exp(dvac.iloc[-1].current_mean - dvac.iloc[0].current_mean) - 1.0
            ),
        },
        "resources": resource_summary(resources),
    }
    (OUT / "SUMMARY_G49.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
