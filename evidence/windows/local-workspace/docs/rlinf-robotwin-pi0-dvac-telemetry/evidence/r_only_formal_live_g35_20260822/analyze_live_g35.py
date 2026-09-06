from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
RAW = HERE / "raw"
OUT = HERE / "analysis"
BASE_SCRIPT = (
    WORKSPACE
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/"
    "r_only_formal_live_g23_20260822/analysis/analyze_r_only_g23.py"
)
BASELINE_JSON = WORKSPACE / "audits/20260717-084926-grpo-current/analysis.json"


def load_base():
    spec = importlib.util.spec_from_file_location("g23_plot_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mean_between(frame: pd.DataFrame, column: str, lo: int, hi: int, step: str) -> float:
    return float(frame.loc[frame[step].between(lo, hi), column].mean())


def main() -> None:
    base = load_base()
    OUT.mkdir(parents=True, exist_ok=True)

    current = base.parse_metrics(RAW / "run/metrics.log")
    rank_paths = sorted((RAW / "run/dvac_train").glob("actor_rank*/runner_step_metrics.csv"))
    npz_paths = sorted((RAW / "run/dvac_train").glob("actor_rank*/rollout_step0034.npz"))
    rank, dvac = base.aggregate_dvac_steps(rank_paths)
    horizon, latest_h = base.latest_horizon(npz_paths)
    resources = pd.read_csv(RAW / "runtime/resource_monitor/resources.csv")
    baseline_payload = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    baseline = pd.DataFrame(baseline_payload["metrics"])

    latest_step = int(current.global_step.max())
    baseline_overlap = baseline[baseline.step <= latest_step].copy()
    merged = current.merge(
        baseline_overlap,
        left_on="global_step",
        right_on="step",
        suffixes=("_r", "_g"),
    )
    x = merged.global_step.to_numpy(float)

    image, draw = base.canvas(
        f"R-only formal live snapshot through Global Step {latest_step}",
        "Training-rollout metrics and method diagnostics; success is not held-out evaluation.",
        1885,
    )
    boxes = base.panel_boxes(4)
    base.draw_chart(
        draw,
        boxes[0],
        "Training-rollout success (raw)",
        x,
        [
            ("R-only", 100 * merged.success_once_r.to_numpy(float), base.ORANGE),
            ("historical GRPO", 100 * merged.success_once_g.to_numpy(float), base.BLUE),
        ],
        "%",
    )
    base.draw_chart(
        draw,
        boxes[1],
        "Training-rollout success (5-step mean)",
        x,
        [
            (
                "R-only",
                100 * merged.success_once_r.rolling(5, min_periods=1).mean().to_numpy(),
                base.ORANGE,
            ),
            (
                "historical GRPO",
                100 * merged.success_once_g.rolling(5, min_periods=1).mean().to_numpy(),
                base.BLUE,
            ),
        ],
        "%",
    )
    for box, key, title, label, scale, reference in (
        (boxes[2], "actor/approx_kl", "Approx KL", "fraction", 1.0, None),
        (boxes[3], "actor/clip_fraction", "PPO joint-query clip fraction", "%", 100.0, None),
        (boxes[4], "actor/grad_norm", "Pre-global-clip gradient norm", "norm", 1.0, 1.0),
    ):
        base.draw_chart(
            draw,
            box,
            title,
            x,
            [
                ("R-only", scale * merged[f"{key}_r"].to_numpy(float), base.ORANGE),
                ("historical GRPO", scale * merged[f"{key}_g"].to_numpy(float), base.BLUE),
            ],
            label,
            reference,
        )
    apply = dvac[dvac.warmup == 0]
    dx = apply.global_step.to_numpy(float)
    base.draw_chart(
        draw,
        boxes[5],
        "Per-action gradient weights",
        dx,
        [
            ("p05", apply.weight_p05.to_numpy(), base.BLUE),
            ("median", apply.weight_p50.to_numpy(), base.PURPLE),
            ("mean", apply.weight_mean.to_numpy(), base.GREEN),
            ("p95", apply.weight_p95.to_numpy(), base.ORANGE),
        ],
        "weight",
        1.0,
    )
    base.draw_chart(
        draw,
        boxes[6],
        "Weight direction and residual clipping",
        dx,
        [
            ("downweighted", 100 * apply.weight_below_one_fraction.to_numpy(), base.BLUE),
            ("upweighted", 100 * apply.weight_above_one_fraction.to_numpy(), base.ORANGE),
            ("at -2", 100 * apply.z_low_clip_fraction.to_numpy(), base.PURPLE),
            ("at +2", 100 * apply.z_high_clip_fraction.to_numpy(), base.GOLD),
        ],
        "% of valid points",
    )
    base.draw_chart(
        draw,
        boxes[7],
        f"Step {latest_step} weights by future-action index",
        horizon.h.to_numpy(float),
        [
            ("p05", horizon.weight_p05.to_numpy(), base.BLUE),
            ("mean", horizon.weight_mean.to_numpy(), base.ORANGE),
            ("p95", horizon.weight_p95.to_numpy(), base.GOLD),
        ],
        "weight",
        1.0,
    )
    image.save(OUT / "TRAIN_AND_METHOD_G35.png", optimize=True)

    resource_image, resource_draw = base.canvas(
        "R-only formal resource telemetry through Global Step 35",
        "The observer records only; cgroup limit is 240 GiB and all memory-event counters remain zero.",
        1035,
    )
    resource_boxes = base.panel_boxes(2)
    resources = resources.sort_values(["elapsed_s", "gpu_index"])
    gpu0 = resources[resources.gpu_index == 0]
    gpu1 = resources[resources.gpu_index == 1]
    time = resources.drop_duplicates("elapsed_s", keep="last")
    base.draw_chart(
        resource_draw,
        resource_boxes[0],
        "GPU memory",
        gpu0.elapsed_s.to_numpy(float) / 3600,
        [
            ("GPU 0", gpu0.gpu_memory_used_mib.to_numpy(float) / 1024, base.ORANGE),
            ("GPU 1", gpu1.gpu_memory_used_mib.to_numpy(float) / 1024, base.BLUE),
        ],
        "GiB",
    )
    base.draw_chart(
        resource_draw,
        resource_boxes[1],
        "GPU utilization",
        gpu0.elapsed_s.to_numpy(float) / 3600,
        [
            ("GPU 0", gpu0.gpu_util_pct.to_numpy(float), base.ORANGE),
            ("GPU 1", gpu1.gpu_util_pct.to_numpy(float), base.BLUE),
        ],
        "%",
    )
    base.draw_chart(
        resource_draw,
        resource_boxes[2],
        "Cgroup RAM",
        time.elapsed_s.to_numpy(float) / 3600,
        [
            ("current", time.cgroup_current_bytes.to_numpy(float) / 2**30, base.ORANGE),
            ("anonymous", time.cgroup_anon_bytes.to_numpy(float) / 2**30, base.PURPLE),
            ("file/cache", time.cgroup_file_bytes.to_numpy(float) / 2**30, base.BLUE),
        ],
        "GiB",
        240.0,
    )
    base.draw_chart(
        resource_draw,
        resource_boxes[3],
        "Available host resources",
        time.elapsed_s.to_numpy(float) / 3600,
        [
            ("disk", time.disk_available_kib.to_numpy(float) / 2**20, base.GREEN),
            ("host RAM", time.host_mem_available_kib.to_numpy(float) / 2**20, base.GOLD),
        ],
        "GiB",
    )
    resource_image.save(OUT / "RESOURCES_G35.png", optimize=True)

    current.to_csv(OUT / "TRAIN_METRICS_G35.csv", index=False)
    rank.to_csv(OUT / "DVAC_RANK_METRICS_G35.csv", index=False)
    dvac.to_csv(OUT / "DVAC_STEP_METRICS_G35.csv", index=False)
    horizon.to_csv(OUT / "LATEST_HORIZON_G35.csv", index=False)

    latest = current.iloc[-1]
    latest5_lo = max(1, latest_step - 4)
    latest10_lo = max(1, latest_step - 9)
    summary = {
        "snapshot": {
            "live_probe_time": "2026-08-22T15:01:25+08:00",
            "latest_complete_global_step": latest_step,
            "next_phase_at_probe": "Global Step 36 rollout 8/16",
            "processes": "wrapper/driver/observer alive",
            "elapsed_text": str(latest.elapsed_text),
            "eta_text": str(latest.eta_text),
        },
        "training_rollout_success": {
            "r_only_mean_g1_to_latest": float(current.success_once.mean()),
            "grpo_mean_same_steps": float(baseline_overlap.success_once.mean()),
            "latest": float(latest.success_once),
            "grpo_same_step": float(baseline_overlap.iloc[-1].success_once),
            "r_only_recent5": mean_between(current, "success_once", latest5_lo, latest_step, "global_step"),
            "grpo_recent5": mean_between(baseline, "success_once", latest5_lo, latest_step, "step"),
            "r_only_recent10": mean_between(current, "success_once", latest10_lo, latest_step, "global_step"),
            "grpo_recent10": mean_between(baseline, "success_once", latest10_lo, latest_step, "step"),
        },
        "optimization_mean_g1_to_latest": {
            "r_only_approx_kl": float(current["actor/approx_kl"].mean()),
            "grpo_approx_kl": float(baseline_overlap["actor/approx_kl"].mean()),
            "r_only_clip_fraction": float(current["actor/clip_fraction"].mean()),
            "grpo_clip_fraction": float(baseline_overlap["actor/clip_fraction"].mean()),
            "r_only_preclip_grad_norm": float(current["actor/grad_norm"].mean()),
            "grpo_preclip_grad_norm": float(baseline_overlap["actor/grad_norm"].mean()),
            "r_only_step_minutes": float(current.step_time_s.mean() / 60),
            "grpo_step_minutes": float(baseline_overlap.step_time_s.mean() / 60),
        },
        "dvac_apply_g2_to_latest": {
            "weight_mean": float(apply.weight_mean.mean()),
            "p05_mean": float(apply.weight_p05.mean()),
            "median_mean": float(apply.weight_p50.mean()),
            "p95_mean": float(apply.weight_p95.mean()),
            "downweighted_fraction": float(apply.weight_below_one_fraction.mean()),
            "upweighted_fraction": float(apply.weight_above_one_fraction.mean()),
            "lower_clip_fraction": float(apply.z_low_clip_fraction.mean()),
            "upper_clip_fraction": float(apply.z_high_clip_fraction.mean()),
            "positive_adv_weight_mean": float(apply.positive_adv_weight_mean.mean()),
            "negative_adv_weight_mean": float(apply.negative_adv_weight_mean.mean()),
            "latest_npz": latest_h,
        },
        "resources": {
            "gpu0_peak_gib": float(gpu0.gpu_memory_used_mib.max() / 1024),
            "gpu1_peak_gib": float(gpu1.gpu_memory_used_mib.max() / 1024),
            "gpu0_latest_gib": float(gpu0.iloc[-1].gpu_memory_used_mib / 1024),
            "gpu1_latest_gib": float(gpu1.iloc[-1].gpu_memory_used_mib / 1024),
            "cgroup_peak_gib": float(time.cgroup_current_bytes.max() / 2**30),
            "cgroup_latest_gib": float(time.iloc[-1].cgroup_current_bytes / 2**30),
            "cgroup_limit_gib": 240.0,
            "host_ram_available_min_gib": float(time.host_mem_available_kib.min() / 2**20),
            "disk_available_min_gib": float(time.disk_available_kib.min() / 2**20),
            "event_low_max": int(time.event_low.max()),
            "event_high_max": int(time.event_high.max()),
            "event_max_max": int(time.event_max.max()),
            "event_oom_max": int(time.event_oom.max()),
            "event_oom_kill_max": int(time.event_oom_kill.max()),
        },
        "remote_artifacts_at_probe": {
            "run_size_reported": "30G",
            "runtime_size_reported": "93M",
            "checkpoints": ["global_step_10", "global_step_20", "global_step_30"],
            "checkpoint_size_each_reported": "9.7G",
            "dvac_npz_count": 70,
            "control_trace_files": 4,
            "control_trace_video_bytes": 15354,
            "control_trace_frame_rows": 115,
            "disk_available_reported": "727G",
        },
    }
    (OUT / "SUMMARY_G35.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
