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
PREVIOUS = (
    WORKSPACE
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/"
    "global_z_w0to2_live_g35_20260824/analyze_global_z_g35.py"
)
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
HISTORICAL_METRICS = (
    WORKSPACE
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/"
    "v3_formal_stop_g49_20260823/analysis/FOUR_RUN_TRAINING_G49.csv"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rename_plot(old_name: str, new_name: str) -> None:
    old = OUT / old_name
    new = OUT / new_name
    if new.exists():
        new.unlink()
    old.replace(new)


def artifact_contract(npz_paths: list[Path], rank_paths: list[Path]) -> dict:
    npz_schema = {}
    for path in npz_paths:
        with np.load(path) as data:
            npz_schema[path.parent.name] = {
                key: {"shape": list(data[key].shape), "dtype": str(data[key].dtype)}
                for key in data.files
            }
    csv_contract = {}
    for path in rank_paths:
        frame = pd.read_csv(path)
        csv_contract[path.parent.name] = {
            "rows": int(len(frame)),
            "columns": frame.columns.tolist(),
            "runner_step_min": int(frame.runner_step.min()),
            "runner_step_max": int(frame.runner_step.max()),
        }
    frames = pd.read_csv(RAW / "run/control_trace/reset_57/frames.csv")
    return {
        "snapshot_file_count": len([path for path in RAW.rglob("*") if path.is_file()]),
        "snapshot_total_bytes": sum(path.stat().st_size for path in RAW.rglob("*") if path.is_file()),
        "npz": npz_schema,
        "runner_step_csv": csv_contract,
        "control_trace": {
            "rows": int(len(frames)),
            "columns": frames.columns.tolist(),
            "video_bytes": int((RAW / "run/control_trace/reset_57/head_camera.mp4").stat().st_size),
            "metadata": json.loads((RAW / "run/control_trace/reset_57/metadata.json").read_text(encoding="utf-8")),
        },
        "other_files": {
            str(path.relative_to(RAW)): path.stat().st_size
            for path in sorted(RAW.rglob("*"))
            if path.is_file() and path.suffix.lower() not in {".npz", ".csv", ".mp4"}
        },
    }


def main() -> None:
    previous = load_module("global_z_g44_previous", PREVIOUS)
    base = load_module("global_z_g44_base", BASE_ANALYSIS)
    plot = load_module("global_z_g44_plot", PLOT_ANALYSIS)
    OUT.mkdir(parents=True, exist_ok=True)
    previous.OUT = OUT

    main_metrics = base.parse_metrics(RAW / "run/metrics.log")
    latest = int(main_metrics.global_step.max())
    if latest != 44:
        raise RuntimeError(f"snapshot contract expected g44, parsed g{latest}")
    current = previous.standardize_current(main_metrics)
    historical = pd.read_csv(HISTORICAL_METRICS)
    historical = historical[historical.step <= latest].copy()
    metrics = pd.concat([historical, current], ignore_index=True, sort=False)
    metrics["success_roll5_pct"] = metrics.groupby("run", sort=False).success_pct.transform(
        lambda series: series.rolling(5, min_periods=1).mean()
    )

    rank_paths = sorted((RAW / "run/dvac_train").glob("actor_rank*/runner_step_metrics.csv"))
    rank, dvac = base.aggregate_dvac_steps(rank_paths)
    npz_paths = sorted((RAW / "run/dvac_train").glob("actor_rank*/rollout_step0043.npz"))
    _, horizon, method = previous.load_latest(npz_paths)
    resources = pd.read_csv(RAW / "runtime/resource_monitor/resources.csv")

    previous.plot_success(plot, metrics, latest)
    rename_plot("GLOBAL_Z_FIVE_RUN_SUCCESS_G35.png", "GLOBAL_Z_FIVE_RUN_SUCCESS_G44.png")
    previous.plot_optimization(plot, metrics, latest)
    rename_plot("GLOBAL_Z_FIVE_RUN_OPTIMIZATION_G35.png", "GLOBAL_Z_FIVE_RUN_OPTIMIZATION_G44.png")
    previous.plot_method(plot, dvac, horizon, latest, method)
    rename_plot("GLOBAL_Z_METHOD_DIAGNOSTICS_G35.png", "GLOBAL_Z_METHOD_DIAGNOSTICS_G44.png")
    previous.plot_resources(plot, resources)
    rename_plot("GLOBAL_Z_RESOURCES_G35.png", "GLOBAL_Z_RESOURCES_G44.png")

    metrics.to_csv(OUT / "FIVE_RUN_TRAINING_G44.csv", index=False)
    rank.to_csv(OUT / "GLOBAL_Z_DVAC_RANK_METRICS_G44.csv", index=False)
    dvac.to_csv(OUT / "GLOBAL_Z_DVAC_STEP_METRICS_G44.csv", index=False)
    horizon.to_csv(OUT / "GLOBAL_Z_LATEST_HORIZON_G44.csv", index=False)
    resources.to_csv(OUT / "GLOBAL_Z_RESOURCES_G44.csv", index=False)

    run_summaries = {
        label: previous.run_summary(metrics[metrics.run == label])
        for label in metrics.run.drop_duplicates().tolist()
    }
    current_summary = run_summaries["DVAC global-z [0,2]"]
    comparisons = {}
    for label in ("Original GRPO", "DVAC v1 global-z", "DVAC v2 R-only", "DVAC v3 R-only [0,2]"):
        other = run_summaries[label]
        comparisons[label] = {
            "mean_success_pp": current_summary["mean_success_pct"] - other["mean_success_pct"],
            "latest5_success_pp": current_summary["latest5_success_pct"] - other["latest5_success_pct"],
            "latest10_success_pp": current_summary["latest10_success_pct"] - other["latest10_success_pct"],
        }

    apply = dvac[dvac.warmup == 0]
    time = resources.drop_duplicates("elapsed_s", keep="last").sort_values("elapsed_s")
    gpu0 = resources[resources.gpu_index == 0]
    gpu1 = resources[resources.gpu_index == 1]
    driver_text = (RAW / "runtime/driver.log").read_text(encoding="utf-8", errors="replace")
    fatal_pattern = re.compile(r"ActorDiedError|RayActorError|CUDA out of memory|OutOfMemory|NCCL.*(?:error|Error)|fatal|FATAL")
    numeric = main_metrics.select_dtypes(include=[np.number]).to_numpy(float)
    contract = artifact_contract(npz_paths, rank_paths)
    summary = {
        "snapshot": {
            "latest_complete_global_step": latest,
            "fatal_matches": len(fatal_pattern.findall(driver_text)),
            "known_optional_curobo_import_tracebacks": driver_text.count("No module named 'curobo'"),
            "all_main_numeric_finite": bool(np.isfinite(numeric).all()),
            "latest_npz_rank_count": len(npz_paths),
        },
        "five_run_training_g1_to_g44": run_summaries,
        "current_minus_comparators": comparisons,
        "global_z_method_latest_g44": method,
        "global_z_method_apply_g2_to_g44": {
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
            "raw_v_geometric_change_g1_to_g44": float(math.exp(dvac.iloc[-1].current_mean - dvac.iloc[0].current_mean) - 1.0),
        },
        "resources": {
            "through_timestamp": str(time.iloc[-1].timestamp),
            "elapsed_hours": float(time.iloc[-1].elapsed_s / 3600),
            "gpu0_peak_gib": float(gpu0.gpu_memory_used_mib.max() / 1024),
            "gpu1_peak_gib": float(gpu1.gpu_memory_used_mib.max() / 1024),
            "gpu0_latest_gib": float(gpu0.iloc[-1].gpu_memory_used_mib / 1024),
            "gpu1_latest_gib": float(gpu1.iloc[-1].gpu_memory_used_mib / 1024),
            "cgroup_start_gib": float(time.iloc[0].cgroup_current_bytes / 2**30),
            "cgroup_latest_gib": float(time.iloc[-1].cgroup_current_bytes / 2**30),
            "cgroup_peak_gib": float(time.cgroup_current_bytes.max() / 2**30),
            "cgroup_limit_gib": float(time.iloc[-1].cgroup_max_bytes / 2**30) if "cgroup_max_bytes" in time.columns else 240.0,
            "anon_latest_gib": float(time.iloc[-1].cgroup_anon_bytes / 2**30),
            "file_latest_gib": float(time.iloc[-1].cgroup_file_bytes / 2**30),
            "event_max_delta": int(time.iloc[-1].event_max - time.iloc[0].event_max),
            "event_oom_delta": int(time.iloc[-1].event_oom - time.iloc[0].event_oom),
            "event_oom_kill_delta": int(time.iloc[-1].event_oom_kill - time.iloc[0].event_oom_kill),
            "host_available_latest_gib": float(time.iloc[-1].host_mem_available_kib / 2**20),
            "disk_available_latest_gib": float(time.iloc[-1].disk_available_kib / 2**20),
        },
        "artifact_contract": contract,
    }
    (OUT / "ARTIFACT_CONTENTS_G44.json").write_text(json.dumps(contract, indent=2, allow_nan=True), encoding="utf-8")
    (OUT / "SUMMARY_G44.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
