from __future__ import annotations

import importlib.util
import json
import math
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
HISTORICAL_METRICS = (
    WORKSPACE
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/"
    "r_only_formal_stop_g51_20260822/analysis/THREE_RUN_METRICS_G51.csv"
)
V2_DVAC_METRICS = (
    WORKSPACE
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/"
    "r_only_formal_stop_g51_20260822/analysis/V2_DVAC_STEP_METRICS_G51.csv"
)
V2_RESOURCES = (
    WORKSPACE
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/"
    "r_only_formal_stop_g51_20260822/raw/runtime/resource_monitor/resources.csv"
)
V2_NPZ_ROOT = (
    WORKSPACE
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/"
    "r_only_formal_stop_g51_20260822/raw/run/dvac_train"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_ess(weights: np.ndarray) -> np.ndarray:
    numerator = np.square(weights.sum(axis=1))
    denominator = weights.shape[1] * np.square(weights).sum(axis=1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def top_mass(values: np.ndarray, fraction: float) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    flat = flat[np.isfinite(flat) & (flat >= 0)]
    if flat.size == 0 or flat.sum() == 0:
        return float("nan")
    ordered = np.sort(flat)[::-1]
    count = max(1, int(math.ceil(fraction * ordered.size)))
    return float(ordered[:count].sum() / ordered.sum())


def global_ess(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    flat = flat[np.isfinite(flat)]
    denominator = flat.size * np.square(flat).sum()
    return float(np.square(flat.sum()) / denominator) if denominator > 0 else 0.0


def load_latest_npz(paths: list[Path]) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, float]]:
    gathered: dict[str, list[np.ndarray]] = {
        "weights": [],
        "residual": [],
        "advantages": [],
        "log_v": [],
    }
    centers: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    for path in paths:
        with np.load(path) as data:
            mask = data["loss_mask"].reshape(-1).astype(bool)
            gathered["weights"].append(data["weights"].reshape(-1, 50)[mask].astype(np.float64))
            gathered["residual"].append(data["residual_z"].reshape(-1, 50)[mask].astype(np.float64))
            gathered["advantages"].append(data["advantages"].reshape(-1)[mask].astype(np.float64))
            gathered["log_v"].append(np.log(data["v_l3"].reshape(-1, 50)[mask].astype(np.float64) + 1e-12))
            centers.append(data["position_center_log_v"].astype(np.float64))
            scales.append(data["position_scale_log_v"].astype(np.float64))
    values = {key: np.concatenate(parts, axis=0) for key, parts in gathered.items()}
    center = np.mean(np.stack(centers), axis=0)
    scale = np.mean(np.stack(scales), axis=0)
    weights = values["weights"]
    residual = values["residual"]
    advantages = values["advantages"]
    query_mean_weight = weights.mean(axis=1)
    ess = safe_ess(weights)
    angle = np.degrees(np.arccos(np.clip(np.sqrt(ess), 0.0, 1.0)))
    uniform_credit = np.repeat(np.abs(advantages)[:, None], 50, axis=1)
    weighted_credit = uniform_credit * weights
    rows: list[dict[str, float | int]] = []
    for h in range(50):
        w = weights[:, h]
        r = residual[:, h]
        rows.append(
            {
                "h": h,
                "position_center_log_v": float(center[h]),
                "position_scale_log_v": float(scale[h]),
                "current_log_v_median": float(np.median(values["log_v"][:, h])),
                "residual_p05": float(np.quantile(r, 0.05)),
                "residual_median": float(np.median(r)),
                "residual_p95": float(np.quantile(r, 0.95)),
                "weight_p05": float(np.quantile(w, 0.05)),
                "weight_mean": float(np.mean(w)),
                "weight_p50": float(np.quantile(w, 0.50)),
                "weight_p95": float(np.quantile(w, 0.95)),
                "at_min_fraction": float(np.mean(w <= 1e-6)),
                "at_max_fraction": float(np.mean(w >= 1.999999)),
            }
        )
    horizon = pd.DataFrame(rows)
    flat = weights.reshape(-1)
    summary = {
        "valid_queries": int(weights.shape[0]),
        "valid_weight_points": int(flat.size),
        "weight_min": float(flat.min()),
        "weight_p05": float(np.quantile(flat, 0.05)),
        "weight_p50": float(np.quantile(flat, 0.50)),
        "weight_mean": float(flat.mean()),
        "weight_p95": float(np.quantile(flat, 0.95)),
        "weight_max": float(flat.max()),
        "weight_below_one_fraction": float(np.mean(flat < 1.0)),
        "weight_above_one_fraction": float(np.mean(flat > 1.0)),
        "weight_at_min_fraction": float(np.mean(flat <= 1e-6)),
        "weight_at_max_fraction": float(np.mean(flat >= 1.999999)),
        "positive_adv_query_weight_mean": float(np.mean(query_mean_weight[advantages > 0])),
        "negative_adv_query_weight_mean": float(np.mean(query_mean_weight[advantages < 0])),
        "per_query_weight_ess_mean": float(np.mean(ess)),
        "effective_h_mean": float(50.0 * np.mean(ess)),
        "coefficient_angle_deg_mean": float(np.mean(angle)),
        "weight_top20_mass": top_mass(weights, 0.20),
        "uniform_credit_top20_mass": top_mass(uniform_credit, 0.20),
        "weighted_credit_top20_mass": top_mass(weighted_credit, 0.20),
        "uniform_global_credit_ess": global_ess(uniform_credit),
        "weighted_global_credit_ess": global_ess(weighted_credit),
        "absolute_credit_mass_ratio": float(weighted_credit.sum() / uniform_credit.sum()),
        "back25_minus_front25_weight": float(weights[:, 25:].mean() - weights[:, :25].mean()),
        "center_rank_max_abs_difference": float(np.max(np.abs(centers[0] - centers[1]))),
        "scale_rank_max_abs_difference": float(np.max(np.abs(scales[0] - scales[1]))),
    }
    return values, horizon, summary


def standardized_v3(metrics: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "run": "DVAC v3 R-only [0,2]",
            "step": metrics.global_step.astype(int),
            "success_pct": 100.0 * metrics.success_once.astype(float),
            "approx_kl": metrics["actor/approx_kl"].astype(float),
            "clip_pct": 100.0 * metrics["actor/clip_fraction"].astype(float),
            "grad_norm": metrics["actor/grad_norm"].astype(float),
            "step_min": metrics.step_time_s.astype(float) / 60.0,
        }
    )
    result["success_roll5_pct"] = result.success_pct.rolling(5, min_periods=1).mean()
    return result


def plot_training(plot, all_metrics: pd.DataFrame, latest: int) -> None:
    colors = {
        "Original GRPO": plot.BLUE,
        "DVAC v1 global-z": plot.GOLD,
        "DVAC v2 R-only": plot.PURPLE,
        "DVAC v3 R-only [0,2]": plot.ORANGE,
    }
    image, draw = plot.make_canvas(
        f"DVAC v3 early training through Global Step {latest}",
        "Step 1 is uniform warmup; Step 2 is the first update using R-only weights in [0,2]. Success is on-policy rollout success.",
        1120,
    )
    specs = [
        ("success_pct", "A. Training-rollout success", "%"),
        ("approx_kl", "B. Approx KL", "fraction"),
        ("clip_pct", "C. PPO joint-query clip fraction", "% queries"),
        ("grad_norm", "D. Pre-global-clip gradient norm", "norm"),
        ("step_min", "E. Global-step wall time", "minutes"),
    ]
    boxes = [
        (45, 135, 885, 590),
        (915, 135, 1755, 590),
        (45, 625, 600, 1080),
        (625, 625, 1180, 1080),
        (1205, 625, 1755, 1080),
    ]
    for index, (box, (key, title, ylabel)) in enumerate(zip(boxes, specs)):
        series = []
        for label in colors:
            frame = all_metrics[all_metrics.run == label].sort_values("step")
            series.append((label, frame.step.to_numpy(float), frame[key].to_numpy(float), colors[label]))
        plot.draw_chart(
            draw,
            box,
            title,
            series,
            ylabel,
            x_limits=(1, latest),
            reference=1.0 if key == "grad_norm" else None,
            legend=index == 0,
        )
    image.save(OUT / "V3_EARLY_TRAINING_G2.png", optimize=True)


def plot_method(plot, dvac: pd.DataFrame, values: dict[str, np.ndarray], horizon: pd.DataFrame, latest: int) -> None:
    image, draw = plot.make_canvas(
        f"DVAC v3 method diagnostics through Global Step {latest}",
        "R-only = per-h robust residual. The [0,2] map changes backward contribution while PPO forward ratio/clip stay joint-query.",
        1120,
    )
    boxes = [(45, 135, 885, 590), (915, 135, 1755, 590), (45, 625, 885, 1080), (915, 625, 1755, 1080)]
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
    ordered = np.sort(values["weights"].reshape(-1))
    cdf = 100.0 * (np.arange(ordered.size) + 1) / ordered.size
    plot.draw_chart(
        draw,
        boxes[1],
        f"B. Step {latest} empirical weight CDF",
        [("all valid action positions", ordered, cdf, plot.ORANGE)],
        "% positions <= weight",
        x_limits=(0, 2),
        reference=None,
    )
    h = horizon.h.to_numpy(float)
    plot.draw_chart(
        draw,
        boxes[2],
        f"C. Step {latest} weights by future-action index",
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
        boxes[3],
        f"D. Step {latest} position-calibrated residual",
        [
            ("p05", h, horizon.residual_p05.to_numpy(float), plot.BLUE),
            ("median", h, horizon.residual_median.to_numpy(float), plot.PURPLE),
            ("p95", h, horizon.residual_p95.to_numpy(float), plot.ORANGE),
        ],
        "residual z",
        x_limits=(0, 49),
        reference=0.0,
    )
    image.save(OUT / "V3_EARLY_METHOD_G2.png", optimize=True)


def plot_resources(plot, resources: pd.DataFrame) -> None:
    resources = resources.drop_duplicates(["elapsed_s", "gpu_index"], keep="last").sort_values(["elapsed_s", "gpu_index"])
    gpu0 = resources[resources.gpu_index == 0]
    gpu1 = resources[resources.gpu_index == 1]
    time = resources.drop_duplicates("elapsed_s", keep="last").sort_values("elapsed_s")
    v2 = pd.read_csv(V2_RESOURCES).drop_duplicates(["elapsed_s", "gpu_index"], keep="last")
    v2_time = v2.drop_duplicates("elapsed_s", keep="last").sort_values("elapsed_s")
    image, draw = plot.make_canvas(
        "DVAC v3 live resource telemetry",
        "Read-only observer data. Cgroup growth is shown relative to each run's own starting point.",
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
        "C. GPU utilization",
        [
            ("GPU 0", gpu0.elapsed_s.to_numpy(float) / 3600, gpu0.gpu_util_pct.to_numpy(float), plot.ORANGE),
            ("GPU 1", gpu1.elapsed_s.to_numpy(float) / 3600, gpu1.gpu_util_pct.to_numpy(float), plot.BLUE),
        ],
        "%",
    )
    plot.draw_chart(
        draw,
        boxes[3],
        "D. Cgroup growth from each run's own start",
        [
            ("v2 R-only", v2_time.elapsed_s.to_numpy(float) / 3600, (v2_time.cgroup_current_bytes.to_numpy(float) - v2_time.cgroup_current_bytes.iloc[0]) / 2**30, plot.PURPLE),
            ("v3 [0,2]", time.elapsed_s.to_numpy(float) / 3600, (time.cgroup_current_bytes.to_numpy(float) - time.cgroup_current_bytes.iloc[0]) / 2**30, plot.ORANGE),
        ],
        "GiB added",
    )
    image.save(OUT / "V3_EARLY_RESOURCES_G2.png", optimize=True)


def main() -> None:
    base = load_module("v3_g2_base", BASE_ANALYSIS)
    plot = load_module("v3_g2_plot", PLOT_ANALYSIS)
    OUT.mkdir(parents=True, exist_ok=True)

    main_metrics = base.parse_metrics(RAW / "run/metrics.log")
    latest = int(main_metrics.global_step.max())
    if latest < 2:
        raise RuntimeError(f"need completed Global Step 2, found {latest}")
    rank_paths = sorted((RAW / "run/dvac_train").glob("actor_rank*/runner_step_metrics.csv"))
    rank, dvac = base.aggregate_dvac_steps(rank_paths)
    npz_paths = sorted((RAW / "run/dvac_train").glob("actor_rank*/rollout_step0001.npz"))
    if len(npz_paths) != 2:
        raise RuntimeError(f"expected two Step 2 NPZs, found {len(npz_paths)}")
    values, horizon, method = load_latest_npz(npz_paths)
    v2_npz_paths = sorted(V2_NPZ_ROOT.glob("actor_rank*/rollout_step0001.npz"))
    if len(v2_npz_paths) != 2:
        raise RuntimeError(f"expected two v2 Step 2 NPZs, found {len(v2_npz_paths)}")
    _, _, v2_method = load_latest_npz(v2_npz_paths)

    historical = pd.read_csv(HISTORICAL_METRICS)
    early = historical[historical.step <= latest].copy()
    v3 = standardized_v3(main_metrics[main_metrics.global_step <= latest])
    all_metrics = pd.concat([early, v3], ignore_index=True, sort=False)

    resources = pd.read_csv(RAW / "runtime/resource_monitor/resources.csv")
    resource_time = resources.drop_duplicates("elapsed_s", keep="last").sort_values("elapsed_s")
    gpu0 = resources[resources.gpu_index == 0]
    gpu1 = resources[resources.gpu_index == 1]
    v2_step2 = pd.read_csv(V2_DVAC_METRICS).query("global_step == 2").iloc[0]
    v3_step2 = dvac.query("global_step == 2").iloc[0]

    plot_training(plot, all_metrics, latest)
    plot_method(plot, dvac[dvac.global_step <= latest], values, horizon, latest)
    plot_resources(plot, resources)

    all_metrics.to_csv(OUT / "FOUR_RUN_EARLY_METRICS_G2.csv", index=False)
    rank.to_csv(OUT / "V3_DVAC_RANK_METRICS_G2.csv", index=False)
    dvac.to_csv(OUT / "V3_DVAC_STEP_METRICS_G2.csv", index=False)
    horizon.to_csv(OUT / "V3_LATEST_HORIZON_G2.csv", index=False)
    resources.to_csv(OUT / "V3_RESOURCES_G2.csv", index=False)
    summary = {
        "snapshot": {
            "latest_complete_global_step": latest,
            "step2_is_first_real_weighted_step": True,
            "checkpoint_count_expected_before_g10": 0,
        },
        "training_latest": {
            "success_pct": float(v3.iloc[-1].success_pct),
            "approx_kl": float(v3.iloc[-1].approx_kl),
            "clip_pct": float(v3.iloc[-1].clip_pct),
            "grad_norm": float(v3.iloc[-1].grad_norm),
            "step_min": float(v3.iloc[-1].step_min),
        },
        "method_step2": method,
        "v2_vs_v3_step2": {
            "v2_weight_p05_p50_p95": [float(v2_step2.weight_p05), float(v2_step2.weight_p50), float(v2_step2.weight_p95)],
            "v3_weight_p05_p50_p95": [float(v3_step2.weight_p05), float(v3_step2.weight_p50), float(v3_step2.weight_p95)],
            "v2_weight_mean": float(v2_step2.weight_mean),
            "v3_weight_mean": float(v3_step2.weight_mean),
            "v2_lower_upper_bound_fraction": [float(v2_step2.z_low_clip_fraction), float(v2_step2.z_high_clip_fraction)],
            "v3_lower_upper_bound_fraction": [float(v3_step2.z_low_clip_fraction), float(v3_step2.z_high_clip_fraction)],
            "v2_per_query_weight_ess": v2_method["per_query_weight_ess_mean"],
            "v3_per_query_weight_ess": method["per_query_weight_ess_mean"],
            "v2_effective_h": v2_method["effective_h_mean"],
            "v3_effective_h": method["effective_h_mean"],
            "v2_coefficient_angle_deg": v2_method["coefficient_angle_deg_mean"],
            "v3_coefficient_angle_deg": method["coefficient_angle_deg_mean"],
            "v2_weight_top20_mass": v2_method["weight_top20_mass"],
            "v3_weight_top20_mass": method["weight_top20_mass"],
        },
        "resources": {
            "through_timestamp": str(resource_time.iloc[-1].timestamp),
            "elapsed_minutes": float(resource_time.iloc[-1].elapsed_s / 60),
            "gpu0_peak_gib": float(gpu0.gpu_memory_used_mib.max() / 1024),
            "gpu1_peak_gib": float(gpu1.gpu_memory_used_mib.max() / 1024),
            "cgroup_start_gib": float(resource_time.iloc[0].cgroup_current_bytes / 2**30),
            "cgroup_latest_gib": float(resource_time.iloc[-1].cgroup_current_bytes / 2**30),
            "cgroup_peak_gib": float(resource_time.cgroup_current_bytes.max() / 2**30),
            "cgroup_growth_gib": float((resource_time.iloc[-1].cgroup_current_bytes - resource_time.iloc[0].cgroup_current_bytes) / 2**30),
            "cgroup_limit_gib": 240.0,
            "event_max_start": int(resource_time.iloc[0].event_max),
            "event_max_latest": int(resource_time.iloc[-1].event_max),
            "event_oom_latest": int(resource_time.iloc[-1].event_oom),
            "event_oom_kill_latest": int(resource_time.iloc[-1].event_oom_kill),
            "host_available_min_gib": float(resource_time.host_mem_available_kib.min() / 2**20),
            "disk_available_min_gib": float(resource_time.disk_available_kib.min() / 2**20),
        },
    }
    (OUT / "SUMMARY_G2.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
