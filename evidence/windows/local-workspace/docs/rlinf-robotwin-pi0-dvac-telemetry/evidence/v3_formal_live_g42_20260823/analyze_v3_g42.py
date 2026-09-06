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
G2_ANALYSIS = (
    WORKSPACE
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/"
    "v3_formal_live_g2_20260822/analyze_v3_g2.py"
)
PLOT_ANALYSIS = (
    WORKSPACE
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/"
    "r_only_formal_live_g50_20260822/analyze_live_g50.py"
)
HISTORICAL_METRICS = (
    WORKSPACE
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/"
    "r_only_formal_stop_g51_20260822/analysis/THREE_RUN_METRICS_G51.csv"
)
V2_RESOURCES = (
    WORKSPACE
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/"
    "r_only_formal_stop_g51_20260822/raw/runtime/resource_monitor/resources.csv"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rolling(values: np.ndarray, window: int = 5) -> np.ndarray:
    return pd.Series(values).rolling(window, min_periods=1).mean().to_numpy(float)


def run_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "steps": int(frame.step.max()),
        "mean_success_pct": float(frame.success_pct.mean()),
        "latest_success_pct": float(frame.iloc[-1].success_pct),
        "latest5_success_pct": float(frame.tail(5).success_pct.mean()),
        "latest10_success_pct": float(frame.tail(10).success_pct.mean()),
        "mean_approx_kl": float(frame.approx_kl.mean()),
        "mean_clip_pct": float(frame.clip_pct.mean()),
        "mean_grad_norm": float(frame.grad_norm.mean()),
        "mean_step_min": float(frame.step_min.mean()),
    }


def plot_training(plot, metrics: pd.DataFrame, latest: int) -> None:
    colors = {
        "Original GRPO": plot.BLUE,
        "DVAC v1 global-z": plot.GOLD,
        "DVAC v2 R-only": plot.PURPLE,
        "DVAC v3 R-only [0,2]": plot.ORANGE,
    }
    image, draw = plot.make_canvas(
        f"DVAC v3 through Global Step {latest} — four-run training view",
        "Success is on-policy training rollout success, not held-out evaluation. Curves use the same global-step axis.",
        1500,
    )
    boxes = [
        (45, 150, 885, 570),
        (915, 150, 1755, 570),
        (45, 610, 885, 1030),
        (915, 610, 1755, 1030),
        (45, 1070, 885, 1460),
        (915, 1070, 1755, 1460),
    ]
    success_series = []
    for label, color in colors.items():
        frame = metrics[metrics.run == label].sort_values("step")
        success_series.append((label, frame.step.to_numpy(float), rolling(frame.success_pct.to_numpy(float), 5), color))
    plot.draw_chart(
        draw,
        boxes[0],
        "A. Training-rollout success — trailing 5-step mean",
        success_series,
        "%",
        x_limits=(1, latest),
        legend=True,
    )
    grpo = metrics[metrics.run == "Original GRPO"].sort_values("step")
    v3 = metrics[metrics.run == "DVAC v3 R-only [0,2]"].sort_values("step")
    delta = rolling(v3.success_pct.to_numpy(float), 5) - rolling(grpo.success_pct.to_numpy(float), 5)
    plot.draw_chart(
        draw,
        boxes[1],
        "B. v3 minus original GRPO — trailing 5-step success",
        [("v3 - GRPO", v3.step.to_numpy(float), delta, plot.ORANGE)],
        "percentage points",
        x_limits=(1, latest),
        reference=0.0,
        legend=True,
    )
    specs = [
        ("approx_kl", "C. Approx KL", "fraction", None),
        ("clip_pct", "D. PPO joint-query clip fraction", "% queries", None),
        ("grad_norm", "E. Pre-global-clip gradient norm", "norm", 1.0),
        ("step_min", "F. Global-step wall time", "minutes", None),
    ]
    for box, (key, title, ylabel, reference) in zip(boxes[2:], specs):
        series = []
        for label, color in colors.items():
            frame = metrics[metrics.run == label].sort_values("step")
            series.append((label, frame.step.to_numpy(float), frame[key].to_numpy(float), color))
        plot.draw_chart(
            draw,
            box,
            title,
            series,
            ylabel,
            x_limits=(1, latest),
            reference=reference,
            legend=False,
        )
    image.save(OUT / "V3_FOUR_RUN_TRAINING_G42.png", optimize=True)


def plot_method(plot, dvac: pd.DataFrame, horizon: pd.DataFrame, latest: int, method: dict[str, float]) -> None:
    image, draw = plot.make_canvas(
        f"DVAC v3 method diagnostics through Global Step {latest}",
        f"Latest per-query weight ESS={method['per_query_weight_ess_mean']:.3f}, effective H={method['effective_h_mean']:.1f}/50, coefficient angle={method['coefficient_angle_deg_mean']:.1f} degrees.",
        1500,
    )
    boxes = [
        (45, 150, 885, 570),
        (915, 150, 1755, 570),
        (45, 610, 885, 1030),
        (915, 610, 1755, 1030),
        (45, 1070, 885, 1460),
        (915, 1070, 1755, 1460),
    ]
    x = dvac.global_step.to_numpy(float)
    plot.draw_chart(
        draw,
        boxes[0],
        "A. Per-action gradient weights",
        [
            ("p05", x, dvac.weight_p05.to_numpy(float), plot.BLUE),
            ("median", x, dvac.weight_p50.to_numpy(float), plot.PURPLE),
            ("mean", x, dvac.weight_mean.to_numpy(float), plot.GREEN),
            ("p95", x, dvac.weight_p95.to_numpy(float), plot.ORANGE),
        ],
        "weight",
        x_limits=(1, latest),
        reference=1.0,
    )
    plot.draw_chart(
        draw,
        boxes[1],
        "B. Weight direction and boundary use",
        [
            ("downweighted", x, 100 * dvac.weight_below_one_fraction.to_numpy(float), plot.BLUE),
            ("upweighted", x, 100 * dvac.weight_above_one_fraction.to_numpy(float), plot.ORANGE),
            ("at 0", x, 100 * dvac.z_low_clip_fraction.to_numpy(float), plot.PURPLE),
            ("at 2", x, 100 * dvac.z_high_clip_fraction.to_numpy(float), plot.GOLD),
        ],
        "% valid points",
        x_limits=(1, latest),
    )
    plot.draw_chart(
        draw,
        boxes[2],
        "C. Mean weight by advantage sign",
        [
            ("positive advantage", x, dvac.positive_adv_weight_mean.to_numpy(float), plot.BLUE),
            ("negative advantage", x, dvac.negative_adv_weight_mean.to_numpy(float), plot.RED),
        ],
        "weight",
        x_limits=(1, latest),
        reference=1.0,
    )
    plot.draw_chart(
        draw,
        boxes[3],
        "D. Current vs recent-history mean log V_L3",
        [
            ("current", x, dvac.current_mean.to_numpy(float), plot.ORANGE),
            ("recent history", x, dvac.history_mean.replace(0, np.nan).to_numpy(float), plot.PURPLE),
        ],
        "mean log V_L3",
        x_limits=(1, latest),
    )
    h = horizon.h.to_numpy(float)
    plot.draw_chart(
        draw,
        boxes[4],
        f"E. Step {latest} weights by future-action index",
        [
            ("p05", h, horizon.weight_p05.to_numpy(float), plot.BLUE),
            ("mean", h, horizon.weight_mean.to_numpy(float), plot.ORANGE),
            ("p95", h, horizon.weight_p95.to_numpy(float), plot.GOLD),
        ],
        "weight",
        x_limits=(0, 49),
        reference=1.0,
    )
    plot.draw_chart(
        draw,
        boxes[5],
        f"F. Step {latest} position-calibrated residual",
        [
            ("p05", h, horizon.residual_p05.to_numpy(float), plot.BLUE),
            ("median", h, horizon.residual_median.to_numpy(float), plot.PURPLE),
            ("p95", h, horizon.residual_p95.to_numpy(float), plot.ORANGE),
        ],
        "residual z",
        x_limits=(0, 49),
        reference=0.0,
    )
    image.save(OUT / "V3_METHOD_DIAGNOSTICS_G42.png", optimize=True)


def plot_resources(plot, resources: pd.DataFrame) -> None:
    resources = resources.drop_duplicates(["elapsed_s", "gpu_index"], keep="last").sort_values(["elapsed_s", "gpu_index"])
    gpu0 = resources[resources.gpu_index == 0]
    gpu1 = resources[resources.gpu_index == 1]
    time = resources.drop_duplicates("elapsed_s", keep="last").sort_values("elapsed_s")
    v2 = pd.read_csv(V2_RESOURCES).drop_duplicates("elapsed_s", keep="last").sort_values("elapsed_s")
    v2 = v2[v2.elapsed_s <= time.elapsed_s.max()]
    image, draw = plot.make_canvas(
        "DVAC v3 live resource telemetry through Global Step 42",
        "The run is alive without OOM, but cgroup memory has touched the 240 GiB ceiling; event_max is observational, not a stop rule.",
        1120,
    )
    boxes = [(45, 135, 885, 590), (915, 135, 1755, 590), (45, 625, 885, 1080), (915, 625, 1755, 1080)]
    plot.draw_chart(
        draw,
        boxes[0],
        "A. GPU memory",
        [
            ("GPU 0", gpu0.elapsed_s.to_numpy(float) / 3600, gpu0.gpu_memory_used_mib.to_numpy(float) / 1024, plot.ORANGE),
            ("GPU 1", gpu1.elapsed_s.to_numpy(float) / 3600, gpu1.gpu_memory_used_mib.to_numpy(float) / 1024, plot.BLUE),
        ],
        "GiB",
    )
    plot.draw_chart(
        draw,
        boxes[1],
        "B. Cgroup memory composition",
        [
            ("total", time.elapsed_s.to_numpy(float) / 3600, time.cgroup_current_bytes.to_numpy(float) / 2**30, plot.ORANGE),
            ("anonymous", time.elapsed_s.to_numpy(float) / 3600, time.cgroup_anon_bytes.to_numpy(float) / 2**30, plot.PURPLE),
            ("file/cache", time.elapsed_s.to_numpy(float) / 3600, time.cgroup_file_bytes.to_numpy(float) / 2**30, plot.BLUE),
        ],
        "GiB",
        reference=240.0,
    )
    plot.draw_chart(
        draw,
        boxes[2],
        "C. Cgroup growth from each run's own start",
        [
            ("v2 R-only", v2.elapsed_s.to_numpy(float) / 3600, (v2.cgroup_current_bytes.to_numpy(float) - v2.cgroup_current_bytes.iloc[0]) / 2**30, plot.PURPLE),
            ("v3 [0,2]", time.elapsed_s.to_numpy(float) / 3600, (time.cgroup_current_bytes.to_numpy(float) - time.cgroup_current_bytes.iloc[0]) / 2**30, plot.ORANGE),
        ],
        "GiB added",
    )
    plot.draw_chart(
        draw,
        boxes[3],
        "D. Memory-limit event counters",
        [
            ("max delta", time.elapsed_s.to_numpy(float) / 3600, time.event_max.to_numpy(float) - time.event_max.iloc[0], plot.RED),
            ("oom", time.elapsed_s.to_numpy(float) / 3600, time.event_oom.to_numpy(float), plot.ORANGE),
            ("oom_kill", time.elapsed_s.to_numpy(float) / 3600, time.event_oom_kill.to_numpy(float), plot.PURPLE),
        ],
        "cumulative count",
        reference=0.0,
    )
    image.save(OUT / "V3_RESOURCES_G42.png", optimize=True)


def main() -> None:
    base = load_module("v3_g42_base", BASE_ANALYSIS)
    g2 = load_module("v3_g42_helpers", G2_ANALYSIS)
    plot = load_module("v3_g42_plot", PLOT_ANALYSIS)
    OUT.mkdir(parents=True, exist_ok=True)

    main_metrics = base.parse_metrics(RAW / "run/metrics.log")
    latest = int(main_metrics.global_step.max())
    if latest != 42:
        raise RuntimeError(f"snapshot contract expected g42, parsed g{latest}")
    v3 = g2.standardized_v3(main_metrics)
    historical = pd.read_csv(HISTORICAL_METRICS)
    metrics = pd.concat([historical[historical.step <= latest], v3], ignore_index=True, sort=False)

    rank_paths = sorted((RAW / "run/dvac_train").glob("actor_rank*/runner_step_metrics.csv"))
    rank, dvac = base.aggregate_dvac_steps(rank_paths)
    npz_paths = sorted((RAW / "run/dvac_train").glob("actor_rank*/rollout_step0041.npz"))
    values, horizon, method = g2.load_latest_npz(npz_paths)
    resources = pd.read_csv(RAW / "runtime/resource_monitor/resources.csv")

    plot_training(plot, metrics, latest)
    plot_method(plot, dvac, horizon, latest, method)
    plot_resources(plot, resources)

    metrics.to_csv(OUT / "FOUR_RUN_TRAINING_G42.csv", index=False)
    rank.to_csv(OUT / "V3_DVAC_RANK_METRICS_G42.csv", index=False)
    dvac.to_csv(OUT / "V3_DVAC_STEP_METRICS_G42.csv", index=False)
    horizon.to_csv(OUT / "V3_LATEST_HORIZON_G42.csv", index=False)
    resources.to_csv(OUT / "V3_RESOURCES_G42.csv", index=False)

    run_summaries = {
        label: run_summary(metrics[metrics.run == label].sort_values("step"))
        for label in metrics.run.unique()
    }
    grpo = run_summaries["Original GRPO"]
    v3_summary = run_summaries["DVAC v3 R-only [0,2]"]
    apply = dvac[dvac.warmup == 0]
    time = resources.drop_duplicates("elapsed_s", keep="last").sort_values("elapsed_s")
    gpu0 = resources[resources.gpu_index == 0]
    gpu1 = resources[resources.gpu_index == 1]
    v2_resource = pd.read_csv(V2_RESOURCES).drop_duplicates("elapsed_s", keep="last").sort_values("elapsed_s")
    target_elapsed = int(time.elapsed_s.max())
    v2_same = v2_resource.iloc[(v2_resource.elapsed_s - target_elapsed).abs().argsort()[:1]].iloc[0]
    first_max = time[time.event_max > time.event_max.iloc[0]]
    driver_text = (RAW / "runtime/driver.log").read_text(encoding="utf-8", errors="replace")
    fatal_pattern = re.compile(r"ActorDiedError|RayActorError|CUDA out of memory|OutOfMemory|NCCL.*(?:error|Error)|fatal|FATAL")
    numeric = main_metrics.select_dtypes(include=[np.number]).to_numpy(float)
    summary = {
        "snapshot": {
            "latest_complete_global_step": latest,
            "next_rollout_started": True,
            "fatal_matches": len(fatal_pattern.findall(driver_text)),
            "all_main_numeric_finite": bool(np.isfinite(numeric).all()),
        },
        "four_run_training_g1_to_g42": run_summaries,
        "v3_minus_original_grpo": {
            "mean_success_pp": v3_summary["mean_success_pct"] - grpo["mean_success_pct"],
            "latest5_success_pp": v3_summary["latest5_success_pct"] - grpo["latest5_success_pct"],
            "latest10_success_pp": v3_summary["latest10_success_pct"] - grpo["latest10_success_pct"],
        },
        "v3_method_g42": method,
        "v3_method_apply_g2_to_g42": {
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
            "raw_log_v_geometric_change_g1_to_g42": float(math.exp(dvac.iloc[-1].current_mean - dvac.iloc[0].current_mean) - 1.0),
        },
        "resources": {
            "through_timestamp": str(time.iloc[-1].timestamp),
            "elapsed_hours": float(time.iloc[-1].elapsed_s / 3600),
            "gpu0_peak_gib": float(gpu0.gpu_memory_used_mib.max() / 1024),
            "gpu1_peak_gib": float(gpu1.gpu_memory_used_mib.max() / 1024),
            "cgroup_start_gib": float(time.iloc[0].cgroup_current_bytes / 2**30),
            "cgroup_latest_gib": float(time.iloc[-1].cgroup_current_bytes / 2**30),
            "cgroup_peak_gib": float(time.cgroup_current_bytes.max() / 2**30),
            "cgroup_limit_gib": 240.0,
            "anon_latest_gib": float(time.iloc[-1].cgroup_anon_bytes / 2**30),
            "file_latest_gib": float(time.iloc[-1].cgroup_file_bytes / 2**30),
            "event_max_start": int(time.iloc[0].event_max),
            "event_max_latest": int(time.iloc[-1].event_max),
            "event_max_delta": int(time.iloc[-1].event_max - time.iloc[0].event_max),
            "first_new_max_event_timestamp": None if first_max.empty else str(first_max.iloc[0].timestamp),
            "first_new_max_event_elapsed_hours": None if first_max.empty else float(first_max.iloc[0].elapsed_s / 3600),
            "event_oom_latest": int(time.iloc[-1].event_oom),
            "event_oom_kill_latest": int(time.iloc[-1].event_oom_kill),
            "v2_same_elapsed_total_gib": float(v2_same.cgroup_current_bytes / 2**30),
            "v2_same_elapsed_anon_gib": float(v2_same.cgroup_anon_bytes / 2**30),
            "v3_minus_v2_same_elapsed_total_gib": float(time.iloc[-1].cgroup_current_bytes / 2**30 - v2_same.cgroup_current_bytes / 2**30),
            "v3_minus_v2_same_elapsed_anon_gib": float(time.iloc[-1].cgroup_anon_bytes / 2**30 - v2_same.cgroup_anon_bytes / 2**30),
            "host_available_min_gib": float(time.host_mem_available_kib.min() / 2**20),
            "disk_available_min_gib": float(time.disk_available_kib.min() / 2**20),
        },
    }
    (OUT / "SUMMARY_G42.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
