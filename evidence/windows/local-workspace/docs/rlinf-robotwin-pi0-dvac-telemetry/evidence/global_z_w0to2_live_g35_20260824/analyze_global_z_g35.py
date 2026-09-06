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
HISTORICAL_METRICS = (
    WORKSPACE
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/"
    "v3_formal_stop_g49_20260823/analysis/FOUR_RUN_TRAINING_G49.csv"
)
V2_RESOURCES = (
    WORKSPACE
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/"
    "r_only_formal_stop_g51_20260822/raw/runtime/resource_monitor/resources.csv"
)
V3_RESOURCES = (
    WORKSPACE
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/"
    "v3_formal_stop_g49_20260823/analysis/V3_RESOURCES_G49.csv"
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
    frame = frame.sort_values("step")
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


def standardize_current(main_metrics: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "run": "DVAC global-z [0,2]",
            "step": main_metrics.global_step.astype(int),
            "success_pct": 100.0 * pd.to_numeric(main_metrics.success_once),
            "approx_kl": pd.to_numeric(main_metrics["actor/approx_kl"]),
            "clip_pct": 100.0 * pd.to_numeric(main_metrics["actor/clip_fraction"]),
            "grad_norm": pd.to_numeric(main_metrics["actor/grad_norm"]),
            "step_min": pd.to_numeric(main_metrics.step_time_s) / 60.0,
        }
    )
    frame["success_roll5_pct"] = frame.success_pct.rolling(5, min_periods=1).mean()
    return frame


def load_latest(npz_paths: list[Path]) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, float]]:
    fields = {key: [] for key in ("v_l2", "v_l3", "v_l4", "weights", "clipped_z")}
    advantages: list[np.ndarray] = []
    formula_errors: list[float] = []
    for path in npz_paths:
        with np.load(path) as data:
            mask = data["loss_mask"].reshape(-1).astype(bool)
            for key in fields:
                fields[key].append(data[key].reshape(-1, 50)[mask].astype(np.float64))
            advantages.append(data["advantages"].reshape(-1)[mask].astype(np.float64))
            formula_errors.append(
                float(
                    np.max(
                        np.abs(
                            data["weights"].astype(np.float64)
                            - (1.0 + 0.5 * data["clipped_z"].astype(np.float64))
                        )
                    )
                )
            )
    values = {key: np.concatenate(parts, axis=0) for key, parts in fields.items()}
    values["advantages"] = np.concatenate(advantages, axis=0)
    values["log_v_l2"] = np.log(values["v_l2"] + 1e-12)
    values["log_v_l3"] = np.log(values["v_l3"] + 1e-12)
    values["log_v_l4"] = np.log(values["v_l4"] + 1e-12)

    rows: list[dict[str, float | int]] = []
    for h in range(50):
        w = values["weights"][:, h]
        z = values["clipped_z"][:, h]
        log_v = values["log_v_l3"][:, h]
        rows.append(
            {
                "h": h,
                "weight_p05": float(np.quantile(w, 0.05)),
                "weight_p50": float(np.quantile(w, 0.50)),
                "weight_mean": float(np.mean(w)),
                "weight_p95": float(np.quantile(w, 0.95)),
                "clipped_z_p05": float(np.quantile(z, 0.05)),
                "clipped_z_p50": float(np.quantile(z, 0.50)),
                "clipped_z_mean": float(np.mean(z)),
                "clipped_z_p95": float(np.quantile(z, 0.95)),
                "log_v_l3_p05": float(np.quantile(log_v, 0.05)),
                "log_v_l3_p50": float(np.quantile(log_v, 0.50)),
                "log_v_l3_mean": float(np.mean(log_v)),
                "log_v_l3_p95": float(np.quantile(log_v, 0.95)),
                "weight_below_one_fraction": float(np.mean(w < 1.0)),
                "weight_above_one_fraction": float(np.mean(w > 1.0)),
                "weight_at_zero_fraction": float(np.mean(w <= 1e-6)),
                "weight_at_two_fraction": float(np.mean(w >= 2.0 - 1e-6)),
            }
        )
    horizon = pd.DataFrame(rows)

    weights = values["weights"]
    adv = values["advantages"]
    denominator = 50.0 * np.square(weights).sum(axis=1)
    ess = np.divide(
        np.square(weights.sum(axis=1)),
        denominator,
        out=np.zeros(weights.shape[0], dtype=np.float64),
        where=denominator > 0,
    )
    coefficient_angle = np.degrees(np.arccos(np.clip(np.sqrt(ess), 0.0, 1.0)))
    pooled_weight = np.sort(weights.reshape(-1))[::-1]
    top20_weight_count = max(1, int(math.ceil(0.20 * pooled_weight.size)))
    uniform_credit = np.abs(np.repeat(adv[:, None], 50, axis=1)).reshape(-1)
    weighted_credit = np.abs(adv[:, None] * weights).reshape(-1)

    def top_mass(vector: np.ndarray, fraction: float) -> float:
        ordered = np.sort(vector)[::-1]
        count = max(1, int(math.ceil(fraction * ordered.size)))
        total = float(ordered.sum())
        return float(ordered[:count].sum() / total) if total > 0 else float("nan")

    def global_ess(vector: np.ndarray) -> float:
        total = float(vector.sum())
        denom = float(vector.size * np.square(vector).sum())
        return float(total * total / denom) if denom > 0 else float("nan")

    positive = adv > 0
    negative = adv < 0
    query_mean_weight = weights.mean(axis=1)
    front = slice(0, 25)
    back = slice(25, 50)
    summary = {
        "valid_queries": int(weights.shape[0]),
        "valid_weight_points": int(weights.size),
        "all_latest_arrays_finite": bool(all(np.isfinite(value).all() for value in values.values())),
        "weight_formula_max_abs_error": float(max(formula_errors)),
        "weight_min": float(weights.min()),
        "weight_p05": float(np.quantile(weights, 0.05)),
        "weight_median": float(np.quantile(weights, 0.50)),
        "weight_mean": float(weights.mean()),
        "weight_p95": float(np.quantile(weights, 0.95)),
        "weight_max": float(weights.max()),
        "downweighted_fraction": float(np.mean(weights < 1.0)),
        "upweighted_fraction": float(np.mean(weights > 1.0)),
        "at_zero_fraction": float(np.mean(weights <= 1e-6)),
        "at_two_fraction": float(np.mean(weights >= 2.0 - 1e-6)),
        "positive_adv_query_weight_mean": float(query_mean_weight[positive].mean()) if positive.any() else float("nan"),
        "negative_adv_query_weight_mean": float(query_mean_weight[negative].mean()) if negative.any() else float("nan"),
        "per_query_weight_ess_mean": float(ess.mean()),
        "effective_h_mean": float(50.0 * ess.mean()),
        "per_query_coefficient_angle_deg_mean": float(coefficient_angle.mean()),
        "pooled_top20_weight_mass": float(pooled_weight[:top20_weight_count].sum() / pooled_weight.sum()),
        "absolute_credit_ratio_vs_uniform": float(weighted_credit.sum() / uniform_credit.sum()),
        "uniform_top10_credit_mass": top_mass(uniform_credit, 0.10),
        "weighted_top10_credit_mass": top_mass(weighted_credit, 0.10),
        "uniform_top20_credit_mass": top_mass(uniform_credit, 0.20),
        "weighted_top20_credit_mass": top_mass(weighted_credit, 0.20),
        "uniform_global_credit_ess": global_ess(uniform_credit),
        "weighted_global_credit_ess": global_ess(weighted_credit),
        "v_l2_median": float(np.median(values["v_l2"])),
        "v_l3_median": float(np.median(values["v_l3"])),
        "v_l4_median": float(np.median(values["v_l4"])),
        "log_v_l3_back_minus_front": float(values["log_v_l3"][:, back].mean() - values["log_v_l3"][:, front].mean()),
        "weight_back_minus_front": float(weights[:, back].mean() - weights[:, front].mean()),
        "weight_h_spearman": float(horizon.h.rank().corr(horizon.weight_mean.rank())),
    }
    return values, horizon, summary


def plot_success(plot, metrics: pd.DataFrame, latest: int) -> None:
    colors = {
        "Original GRPO": plot.BLUE,
        "DVAC v1 global-z": plot.GOLD,
        "DVAC v2 R-only": plot.PURPLE,
        "DVAC v3 R-only [0,2]": plot.ORANGE,
        "DVAC global-z [0,2]": plot.GREEN,
    }
    image, draw = plot.make_canvas(
        f"Five-run training-rollout success through Global Step {latest}",
        "Same global-step axis; this is on-policy training rollout success, not fixed-ID held-out evaluation.",
        1120,
    )
    boxes = [(55, 150, 1745, 590), (55, 630, 1745, 1080)]
    raw: list[tuple[str, np.ndarray, np.ndarray, str]] = []
    roll5: list[tuple[str, np.ndarray, np.ndarray, str]] = []
    for label, color in colors.items():
        frame = metrics[metrics.run == label].sort_values("step")
        raw.append((label, frame.step.to_numpy(float), frame.success_pct.to_numpy(float), color))
        roll5.append((label, frame.step.to_numpy(float), rolling(frame.success_pct.to_numpy(float), 5), color))
    plot.draw_chart(draw, boxes[0], "A. Success at each global step", raw, "%", x_limits=(1, latest))
    plot.draw_chart(draw, boxes[1], "B. Trailing 5-step mean", roll5, "%", x_limits=(1, latest))
    image.save(OUT / "GLOBAL_Z_FIVE_RUN_SUCCESS_G35.png", optimize=True)


def plot_optimization(plot, metrics: pd.DataFrame, latest: int) -> None:
    colors = {
        "Original GRPO": plot.BLUE,
        "DVAC v1 global-z": plot.GOLD,
        "DVAC v2 R-only": plot.PURPLE,
        "DVAC v3 R-only [0,2]": plot.ORANGE,
        "DVAC global-z [0,2]": plot.GREEN,
    }
    image, draw = plot.make_canvas(
        f"Optimization metrics through Global Step {latest}",
        "Global-z [0,2] changes per-h backward credit; PPO ratio and clipping remain joint-query level.",
        1120,
    )
    boxes = [(45, 135, 885, 590), (915, 135, 1755, 590), (45, 625, 885, 1080), (915, 625, 1755, 1080)]
    specs = [
        ("approx_kl", "A. Approx KL", "fraction", None),
        ("clip_pct", "B. PPO joint-query clip fraction", "% queries", None),
        ("grad_norm", "C. Pre-global-clip gradient norm", "norm", 1.0),
        ("step_min", "D. Global-step wall time", "minutes", None),
    ]
    for index, (box, (key, title, ylabel, reference)) in enumerate(zip(boxes, specs)):
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
            legend=index == 0,
        )
    image.save(OUT / "GLOBAL_Z_FIVE_RUN_OPTIMIZATION_G35.png", optimize=True)


def plot_method(plot, dvac: pd.DataFrame, horizon: pd.DataFrame, latest: int, method: dict[str, float]) -> None:
    apply = dvac[dvac.warmup == 0].copy()
    x = apply.global_step.to_numpy(float)
    image, draw = plot.make_canvas(
        f"Global-z [0,2] DVAC diagnostics through Global Step {latest}",
        f"Step 1 is warmup. Latest weight ESS={method['per_query_weight_ess_mean']:.3f}, effective H={method['effective_h_mean']:.1f}/50, coefficient angle={method['per_query_coefficient_angle_deg_mean']:.1f} degrees.",
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
    plot.draw_chart(
        draw,
        boxes[0],
        "A. Per-action gradient weights",
        [
            ("p05", x, apply.weight_p05.to_numpy(float), plot.BLUE),
            ("median", x, apply.weight_p50.to_numpy(float), plot.PURPLE),
            ("mean", x, apply.weight_mean.to_numpy(float), plot.GREEN),
            ("p95", x, apply.weight_p95.to_numpy(float), plot.ORANGE),
        ],
        "weight",
        x_limits=(2, latest),
        reference=1.0,
    )
    plot.draw_chart(
        draw,
        boxes[1],
        "B. Weight direction and boundary use",
        [
            ("downweighted", x, 100 * apply.weight_below_one_fraction.to_numpy(float), plot.BLUE),
            ("upweighted", x, 100 * apply.weight_above_one_fraction.to_numpy(float), plot.ORANGE),
            ("at 0", x, 100 * apply.z_low_clip_fraction.to_numpy(float), plot.PURPLE),
            ("at 2", x, 100 * apply.z_high_clip_fraction.to_numpy(float), plot.GOLD),
        ],
        "% valid points",
        x_limits=(2, latest),
    )
    plot.draw_chart(
        draw,
        boxes[2],
        "C. Mean weight by advantage sign",
        [
            ("positive advantage", x, apply.positive_adv_weight_mean.to_numpy(float), plot.BLUE),
            ("negative advantage", x, apply.negative_adv_weight_mean.to_numpy(float), plot.RED),
        ],
        "weight",
        x_limits=(2, latest),
        reference=1.0,
    )
    plot.draw_chart(
        draw,
        boxes[3],
        "D. Current vs recent-history mean log V_L3",
        [
            ("current", x, apply.current_mean.to_numpy(float), plot.ORANGE),
            ("recent history", x, apply.history_mean.to_numpy(float), plot.PURPLE),
        ],
        "mean log V_L3",
        x_limits=(2, latest),
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
        f"F. Step {latest} global-z by future-action index",
        [
            ("p05", h, horizon.clipped_z_p05.to_numpy(float), plot.BLUE),
            ("median", h, horizon.clipped_z_p50.to_numpy(float), plot.PURPLE),
            ("p95", h, horizon.clipped_z_p95.to_numpy(float), plot.ORANGE),
        ],
        "clipped z",
        x_limits=(0, 49),
        reference=0.0,
    )
    image.save(OUT / "GLOBAL_Z_METHOD_DIAGNOSTICS_G35.png", optimize=True)


def plot_resources(plot, resources: pd.DataFrame) -> None:
    resources = resources.drop_duplicates(["elapsed_s", "gpu_index"], keep="last").sort_values(["elapsed_s", "gpu_index"])
    gpu0 = resources[resources.gpu_index == 0]
    gpu1 = resources[resources.gpu_index == 1]
    time = resources.drop_duplicates("elapsed_s", keep="last").sort_values("elapsed_s")
    v2 = pd.read_csv(V2_RESOURCES).drop_duplicates("elapsed_s", keep="last").sort_values("elapsed_s")
    v3 = pd.read_csv(V3_RESOURCES).drop_duplicates("elapsed_s", keep="last").sort_values("elapsed_s")
    v2 = v2[v2.elapsed_s <= time.elapsed_s.max()]
    v3 = v3[v3.elapsed_s <= time.elapsed_s.max()]
    image, draw = plot.make_canvas(
        "Global-z [0,2] resource telemetry",
        "Observer-only measurements; memory event counters are shown as deltas from this run's first sample.",
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
            ("v3 R-only [0,2]", v3.elapsed_s.to_numpy(float) / 3600, (v3.cgroup_current_bytes.to_numpy(float) - v3.cgroup_current_bytes.iloc[0]) / 2**30, plot.ORANGE),
            ("global-z [0,2]", time.elapsed_s.to_numpy(float) / 3600, (time.cgroup_current_bytes.to_numpy(float) - time.cgroup_current_bytes.iloc[0]) / 2**30, plot.GREEN),
        ],
        "GiB added",
    )
    plot.draw_chart(
        draw,
        boxes[3],
        "D. Memory-limit event counters",
        [
            ("max delta", time.elapsed_s.to_numpy(float) / 3600, time.event_max.to_numpy(float) - time.event_max.iloc[0], plot.RED),
            ("oom delta", time.elapsed_s.to_numpy(float) / 3600, time.event_oom.to_numpy(float) - time.event_oom.iloc[0], plot.ORANGE),
            ("oom-kill delta", time.elapsed_s.to_numpy(float) / 3600, time.event_oom_kill.to_numpy(float) - time.event_oom_kill.iloc[0], plot.PURPLE),
        ],
        "cumulative delta",
        reference=0.0,
    )
    image.save(OUT / "GLOBAL_Z_RESOURCES_G35.png", optimize=True)


def main() -> None:
    base = load_module("global_z_g35_base", BASE_ANALYSIS)
    plot = load_module("global_z_g35_plot", PLOT_ANALYSIS)
    OUT.mkdir(parents=True, exist_ok=True)

    main_metrics = base.parse_metrics(RAW / "run/metrics.log")
    latest = int(main_metrics.global_step.max())
    if latest != 35:
        raise RuntimeError(f"snapshot contract expected g35, parsed g{latest}")
    current = standardize_current(main_metrics)
    historical = pd.read_csv(HISTORICAL_METRICS)
    historical = historical[historical.step <= latest].copy()
    metrics = pd.concat([historical, current], ignore_index=True, sort=False)
    metrics["success_roll5_pct"] = metrics.groupby("run", sort=False).success_pct.transform(
        lambda series: series.rolling(5, min_periods=1).mean()
    )

    rank_paths = sorted((RAW / "run/dvac_train").glob("actor_rank*/runner_step_metrics.csv"))
    rank, dvac = base.aggregate_dvac_steps(rank_paths)
    npz_paths = sorted((RAW / "run/dvac_train").glob("actor_rank*/rollout_step0034.npz"))
    values, horizon, method = load_latest(npz_paths)
    resources = pd.read_csv(RAW / "runtime/resource_monitor/resources.csv")

    plot_success(plot, metrics, latest)
    plot_optimization(plot, metrics, latest)
    plot_method(plot, dvac, horizon, latest, method)
    plot_resources(plot, resources)

    metrics.to_csv(OUT / "FIVE_RUN_TRAINING_G35.csv", index=False)
    rank.to_csv(OUT / "GLOBAL_Z_DVAC_RANK_METRICS_G35.csv", index=False)
    dvac.to_csv(OUT / "GLOBAL_Z_DVAC_STEP_METRICS_G35.csv", index=False)
    horizon.to_csv(OUT / "GLOBAL_Z_LATEST_HORIZON_G35.csv", index=False)
    resources.to_csv(OUT / "GLOBAL_Z_RESOURCES_G35.csv", index=False)

    run_summaries = {
        label: run_summary(metrics[metrics.run == label])
        for label in metrics.run.drop_duplicates().tolist()
    }
    current_summary = run_summaries["DVAC global-z [0,2]"]
    grpo = run_summaries["Original GRPO"]
    v1 = run_summaries["DVAC v1 global-z"]
    v3 = run_summaries["DVAC v3 R-only [0,2]"]
    apply = dvac[dvac.warmup == 0]
    time = resources.drop_duplicates("elapsed_s", keep="last").sort_values("elapsed_s")
    gpu0 = resources[resources.gpu_index == 0]
    gpu1 = resources[resources.gpu_index == 1]
    driver_text = (RAW / "runtime/driver.log").read_text(encoding="utf-8", errors="replace")
    fatal_pattern = re.compile(r"ActorDiedError|RayActorError|CUDA out of memory|OutOfMemory|NCCL.*(?:error|Error)|fatal|FATAL")
    numeric = main_metrics.select_dtypes(include=[np.number]).to_numpy(float)
    summary = {
        "snapshot": {
            "latest_complete_global_step": latest,
            "fatal_matches": len(fatal_pattern.findall(driver_text)),
            "known_optional_curobo_import_tracebacks": driver_text.count("No module named 'curobo'"),
            "all_main_numeric_finite": bool(np.isfinite(numeric).all()),
            "latest_npz_rank_count": len(npz_paths),
        },
        "five_run_training_g1_to_g35": run_summaries,
        "current_minus_original_grpo": {
            "mean_success_pp": current_summary["mean_success_pct"] - grpo["mean_success_pct"],
            "latest5_success_pp": current_summary["latest5_success_pct"] - grpo["latest5_success_pct"],
            "latest10_success_pp": current_summary["latest10_success_pct"] - grpo["latest10_success_pct"],
        },
        "current_minus_v1_same_global_z_signal": {
            "mean_success_pp": current_summary["mean_success_pct"] - v1["mean_success_pct"],
            "latest5_success_pp": current_summary["latest5_success_pct"] - v1["latest5_success_pct"],
            "latest10_success_pp": current_summary["latest10_success_pct"] - v1["latest10_success_pct"],
        },
        "current_minus_v3_same_weight_range": {
            "mean_success_pp": current_summary["mean_success_pct"] - v3["mean_success_pct"],
            "latest5_success_pp": current_summary["latest5_success_pct"] - v3["latest5_success_pct"],
            "latest10_success_pp": current_summary["latest10_success_pct"] - v3["latest10_success_pct"],
        },
        "global_z_method_latest_g35": method,
        "global_z_method_apply_g2_to_g35": {
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
            "raw_v_geometric_change_g1_to_g35": float(math.exp(dvac.iloc[-1].current_mean - dvac.iloc[0].current_mean) - 1.0),
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
    }
    (OUT / "SUMMARY_G35.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
