from __future__ import annotations

import csv
import importlib.util
import json
import math
import re
import zipfile
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


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
BASELINE_RESOURCE = WORKSPACE / "audits/20260717-084926-grpo-current/resources.csv"
V1_ZIP = WORKSPACE / "exports/idea2_dvac_v1_formal_stop_g54_20260821.zip"
V1_METRICS_MEMBER = "idea2_dvac_v1_formal_stop_g54_20260821/analysis/TRAIN_METRICS.csv"
V1_RESOURCE = (
    WORKSPACE
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/"
    "formal_live_step47_20260821/runtime/resources.csv"
)

FONT_REGULAR = Path(r"C:\Windows\Fonts\segoeui.ttf")
FONT_SEMIBOLD = Path(r"C:\Windows\Fonts\seguisb.ttf")
INK = "#17233D"
MUTED = "#5E6C84"
GRID = "#D9E1EA"
WHITE = "#FFFFFF"
BLUE = "#2563EB"
ORANGE = "#EA580C"
PURPLE = "#7C3AED"
GREEN = "#15803D"
GOLD = "#D97706"
RED = "#DC2626"


def font(size: int, semibold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_SEMIBOLD if semibold else FONT_REGULAR), size=size)


def load_base():
    spec = importlib.util.spec_from_file_location("g23_plot_base_for_g51", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rolling(values: np.ndarray, window: int = 5) -> np.ndarray:
    return pd.Series(values).rolling(window, min_periods=1).mean().to_numpy(float)


def standardize(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    if source == "baseline":
        result = pd.DataFrame(
            {
                "step": pd.to_numeric(frame["step"]),
                "success_pct": 100.0 * pd.to_numeric(frame["success_once"]),
                "approx_kl": pd.to_numeric(frame["actor/approx_kl"]),
                "clip_pct": 100.0 * pd.to_numeric(frame["actor/clip_fraction"]),
                "grad_norm": pd.to_numeric(frame["actor/grad_norm"]),
                "step_min": pd.to_numeric(frame["step_time_s"]) / 60.0,
            }
        )
    else:
        result = pd.DataFrame(
            {
                "step": pd.to_numeric(frame["global_step"]),
                "success_pct": 100.0 * pd.to_numeric(frame["success_once"]),
                "approx_kl": pd.to_numeric(frame["actor/approx_kl"]),
                "clip_pct": 100.0 * pd.to_numeric(frame["actor/clip_fraction"]),
                "grad_norm": pd.to_numeric(frame["actor/grad_norm"]),
                "step_min": pd.to_numeric(frame["step_time_s"]) / 60.0,
            }
        )
    return result.drop_duplicates("step", keep="last").sort_values("step").reset_index(drop=True)


def load_runs(base) -> tuple[dict[str, tuple[pd.DataFrame, str]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_payload = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    baseline_raw = pd.DataFrame(baseline_payload["metrics"])
    with zipfile.ZipFile(V1_ZIP) as archive:
        v1_text = archive.read(V1_METRICS_MEMBER).decode("utf-8-sig")
    v1_raw = pd.read_csv(StringIO(v1_text))
    v2_raw = base.parse_metrics(RAW / "run/metrics.log")
    runs = {
        "Original GRPO": (standardize(baseline_raw, "baseline"), BLUE),
        "DVAC v1 global-z": (standardize(v1_raw, "v1"), GOLD),
        "DVAC v2 R-only": (standardize(v2_raw, "v2"), PURPLE),
    }
    return runs, baseline_raw, v1_raw, v2_raw


def draw_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    series: list[tuple[str, np.ndarray, np.ndarray, str]],
    ylabel: str,
    x_limits: tuple[float, float] | None = None,
    y_limits: tuple[float, float] | None = None,
    reference: float | None = None,
    legend: bool = True,
) -> None:
    left, top, right, bottom = box
    draw.text((left, top), title, fill=INK, font=font(27, True))
    plot_left, plot_right = left + 92, right - 24
    plot_top, plot_bottom = top + 60, bottom - 48
    finite = [ys[np.isfinite(ys)] for _, _, ys, _ in series if np.any(np.isfinite(ys))]
    values = np.concatenate(finite)
    if reference is not None:
        values = np.append(values, reference)
    if y_limits is None:
        low, high = float(np.min(values)), float(np.max(values))
        pad = 0.12 * (high - low) if high > low else max(abs(high) * 0.1, 1.0)
        low, high = low - pad, high + pad
    else:
        low, high = y_limits
    if x_limits is None:
        all_x = np.concatenate([xs[np.isfinite(xs)] for _, xs, _, _ in series])
        x_low, x_high = float(np.min(all_x)), float(np.max(all_x))
    else:
        x_low, x_high = x_limits
    if x_high == x_low:
        x_high += 1.0

    def px(value: float) -> float:
        return plot_left + (value - x_low) / (x_high - x_low) * (plot_right - plot_left)

    def py(value: float) -> float:
        return plot_bottom - (value - low) / (high - low) * (plot_bottom - plot_top)

    for tick in range(5):
        value = low + (high - low) * tick / 4
        y = py(value)
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=2)
        label = f"{value:.3f}" if abs(value) < 2 else f"{value:.1f}"
        width = draw.textlength(label, font=font(17))
        draw.text((plot_left - width - 12, y - 10), label, fill=MUTED, font=font(17))
    for tick in range(5):
        value = x_low + (x_high - x_low) * tick / 4
        x = px(value)
        label = f"{value:.1f}" if x_high <= 24 else str(int(round(value)))
        width = draw.textlength(label, font=font(17))
        draw.text((x - width / 2, plot_bottom + 10), label, fill=MUTED, font=font(17))
    if reference is not None and low <= reference <= high:
        y = py(reference)
        for x in range(int(plot_left), int(plot_right), 18):
            draw.line((x, y, min(x + 9, plot_right), y), fill=MUTED, width=2)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=INK, width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=INK, width=2)
    draw.text((left, plot_top), ylabel, fill=MUTED, font=font(17))

    legend_x, legend_y = plot_left, plot_top + 5
    for label, xs, ys, color in series:
        valid = [(float(x), float(y)) for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y)]
        if len(valid) > 1:
            stride = max(1, len(valid) // 1800)
            points = [(px(x), py(y)) for x, y in valid[::stride]]
            if points[-1] != (px(valid[-1][0]), py(valid[-1][1])):
                points.append((px(valid[-1][0]), py(valid[-1][1])))
            draw.line(points, fill=color, width=4, joint="curve")
        if valid:
            x, y = px(valid[-1][0]), py(valid[-1][1])
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
        if legend:
            draw.line((legend_x, legend_y + 11, legend_x + 28, legend_y + 11), fill=color, width=5)
            draw.text((legend_x + 36, legend_y), label, fill=INK, font=font(17))
            legend_x += int(draw.textlength(label, font=font(17))) + 76


def make_canvas(title: str, subtitle: str, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (1800, height), WHITE)
    draw = ImageDraw.Draw(image)
    draw.text((70, 45), title, fill=INK, font=font(43, True))
    draw.text((72, 105), subtitle, fill=MUTED, font=font(22))
    return image, draw


def plot_success(runs: dict[str, tuple[pd.DataFrame, str]], latest: int) -> None:
    image, draw = make_canvas(
        f"Original GRPO vs DVAC v1 vs DVAC v2 — through v2 Global Step {latest}",
        "On-policy training-rollout success; this is not fixed-ID held-out evaluation.",
        1500,
    )
    raw = []
    roll5 = []
    for label, (frame, color) in runs.items():
        x = frame.step.to_numpy(float)
        y = frame.success_pct.to_numpy(float)
        raw.append((label, x, y, color))
        roll5.append((label, x, rolling(y, 5), color))
    draw_chart(draw, (55, 180, 1745, 585), "A. Success at each global step", raw, "%", (1, 100), (65, 102))
    draw_chart(draw, (55, 620, 1745, 1025), "B. Trailing 5-step mean", roll5, "%", (1, 100), (72, 101))

    baseline = runs["Original GRPO"][0]
    baseline_roll = rolling(baseline.success_pct.to_numpy(float), 5)
    deltas = []
    for label in ("DVAC v1 global-z", "DVAC v2 R-only"):
        frame, color = runs[label]
        steps = frame.step.to_numpy(int)
        deltas.append(
            (
                f"{label} minus GRPO",
                steps.astype(float),
                rolling(frame.success_pct.to_numpy(float), 5) - baseline_roll[steps - 1],
                color,
            )
        )
    draw_chart(
        draw,
        (55, 1060, 1745, 1460),
        "C. Trailing-5 difference from original GRPO",
        deltas,
        "percentage points",
        (1, 55),
        reference=0.0,
    )
    image.save(OUT / "THREE_RUN_SUCCESS_G51.png", optimize=True)


def plot_optimization(runs: dict[str, tuple[pd.DataFrame, str]]) -> None:
    image, draw = make_canvas(
        "Three-run optimization metrics",
        "Same training protocol; v1 ends at g54, v2 stopped after complete g51, original GRPO ends at g100.",
        1120,
    )
    specs = [
        ("approx_kl", "A. Approx KL", "fraction"),
        ("clip_pct", "B. PPO joint-query clip fraction", "% queries"),
        ("grad_norm", "C. Pre-global-clip gradient norm", "norm"),
        ("step_min", "D. Global-step wall time", "minutes"),
    ]
    boxes = [(45, 135, 885, 590), (915, 135, 1755, 590), (45, 625, 885, 1080), (915, 625, 1755, 1080)]
    for index, (box, (key, title, ylabel)) in enumerate(zip(boxes, specs)):
        series = [
            (label, frame.step.to_numpy(float), frame[key].to_numpy(float), color)
            for label, (frame, color) in runs.items()
        ]
        draw_chart(draw, box, title, series, ylabel, (1, 100), reference=1.0 if key == "grad_norm" else None, legend=index == 0)
    image.save(OUT / "THREE_RUN_OPTIMIZATION_G51.png", optimize=True)


def latest_weight_geometry(npz_paths: list[Path]) -> dict[str, float]:
    weights_list: list[np.ndarray] = []
    advantages_list: list[np.ndarray] = []
    for path in npz_paths:
        with np.load(path) as data:
            mask = data["loss_mask"].reshape(-1).astype(bool)
            weights_list.append(data["weights"].reshape(-1, 50)[mask].astype(np.float64))
            advantages_list.append(data["advantages"].reshape(-1)[mask].astype(np.float64))
    weights = np.concatenate(weights_list, axis=0)
    advantages = np.concatenate(advantages_list, axis=0)
    ess = np.square(weights.sum(axis=1)) / (50.0 * np.square(weights).sum(axis=1))
    coeff_angle = np.degrees(np.arccos(np.clip(np.sqrt(ess), 0.0, 1.0)))
    c0 = np.repeat(advantages[:, None], 50, axis=1).reshape(-1)
    c1 = (advantages[:, None] * weights).reshape(-1)
    cosine = float(np.dot(c0, c1) / (np.linalg.norm(c0) * np.linalg.norm(c1)))
    pooled = np.sort(weights.reshape(-1))[::-1]
    top20_count = max(1, int(math.ceil(0.20 * pooled.size)))
    return {
        "valid_queries": int(weights.shape[0]),
        "weight_ess_ratio_mean": float(np.mean(ess)),
        "effective_h_mean": float(50.0 * np.mean(ess)),
        "per_query_coefficient_angle_deg_mean": float(np.mean(coeff_angle)),
        "advantage_weighted_coefficient_cosine": cosine,
        "advantage_weighted_coefficient_angle_deg": float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))),
        "pooled_top20_weight_mass": float(pooled[:top20_count].sum() / pooled.sum()),
    }


def plot_method(dvac: pd.DataFrame, horizon: pd.DataFrame, latest: int) -> None:
    apply = dvac[dvac.warmup == 0].copy()
    x = apply.global_step.to_numpy(float)
    image, draw = make_canvas(
        f"R-only DVAC method diagnostics through Global Step {latest}",
        "Step 1 is warmup; subsequent steps use recent-5 per-h robust residual mapped to [0.5, 1.2].",
        1120,
    )
    boxes = [(45, 135, 885, 590), (915, 135, 1755, 590), (45, 625, 885, 1080), (915, 625, 1755, 1080)]
    draw_chart(
        draw,
        boxes[0],
        "A. Per-action gradient weights",
        [
            ("p05", x, apply.weight_p05.to_numpy(float), BLUE),
            ("median", x, apply.weight_p50.to_numpy(float), PURPLE),
            ("mean", x, apply.weight_mean.to_numpy(float), GREEN),
            ("p95", x, apply.weight_p95.to_numpy(float), ORANGE),
        ],
        "weight",
        reference=1.0,
    )
    draw_chart(
        draw,
        boxes[1],
        "B. Mean weight by advantage sign",
        [
            ("positive advantage", x, apply.positive_adv_weight_mean.to_numpy(float), BLUE),
            ("negative advantage", x, apply.negative_adv_weight_mean.to_numpy(float), RED),
        ],
        "weight",
        reference=1.0,
    )
    draw_chart(
        draw,
        boxes[2],
        "C. Weight direction and residual clipping",
        [
            ("downweighted", x, 100 * apply.weight_below_one_fraction.to_numpy(float), BLUE),
            ("upweighted", x, 100 * apply.weight_above_one_fraction.to_numpy(float), ORANGE),
            ("at 0.5", x, 100 * apply.z_low_clip_fraction.to_numpy(float), PURPLE),
            ("at 1.2", x, 100 * apply.z_high_clip_fraction.to_numpy(float), GOLD),
        ],
        "% valid points",
    )
    h = horizon.h.to_numpy(float)
    draw_chart(
        draw,
        boxes[3],
        f"D. Step {latest} weights by future-action index",
        [
            ("p05", h, horizon.weight_p05.to_numpy(float), BLUE),
            ("mean", h, horizon.weight_mean.to_numpy(float), ORANGE),
            ("p95", h, horizon.weight_p95.to_numpy(float), GOLD),
        ],
        "weight",
        reference=1.0,
    )
    image.save(OUT / "V2_METHOD_DIAGNOSTICS_G51.png", optimize=True)


def resource_growth_series() -> list[tuple[str, np.ndarray, np.ndarray, str]]:
    baseline = pd.read_csv(BASELINE_RESOURCE)
    baseline_time = pd.to_datetime(baseline.timestamp)
    baseline_hours = (baseline_time - baseline_time.iloc[0]).dt.total_seconds().to_numpy(float) / 3600.0
    baseline_gib = baseline.cgroup_ram_mb.to_numpy(float) / 1024.0

    v1 = pd.read_csv(V1_RESOURCE).drop_duplicates("elapsed_s", keep="last").sort_values("elapsed_s")
    v1_hours = v1.elapsed_s.to_numpy(float) / 3600.0
    v1_gib = v1.cgroup_current_bytes.to_numpy(float) / 2**30

    v2 = pd.read_csv(RAW / "runtime/resource_monitor/resources.csv").drop_duplicates("elapsed_s", keep="last").sort_values("elapsed_s")
    v2_hours = v2.elapsed_s.to_numpy(float) / 3600.0
    v2_gib = v2.cgroup_current_bytes.to_numpy(float) / 2**30
    return [
        ("Original GRPO", baseline_hours, baseline_gib - baseline_gib[0], BLUE),
        ("DVAC v1", v1_hours, v1_gib - v1_gib[0], GOLD),
        ("DVAC v2", v2_hours, v2_gib - v2_gib[0], PURPLE),
    ]


def plot_resources(resources: pd.DataFrame) -> None:
    resources = resources.sort_values(["elapsed_s", "gpu_index"])
    gpu0 = resources[resources.gpu_index == 0]
    gpu1 = resources[resources.gpu_index == 1]
    time = resources.drop_duplicates("elapsed_s", keep="last").sort_values("elapsed_s")
    image, draw = make_canvas(
        "R-only v2 resource telemetry",
        "Observer-only measurements. The run touched the 240 GiB cgroup ceiling without OOM/OOM-kill.",
        1120,
    )
    boxes = [(45, 135, 885, 590), (915, 135, 1755, 590), (45, 625, 885, 1080), (915, 625, 1755, 1080)]
    draw_chart(
        draw,
        boxes[0],
        "A. GPU memory",
        [
            ("GPU 0", gpu0.elapsed_s.to_numpy(float) / 3600, gpu0.gpu_memory_used_mib.to_numpy(float) / 1024, ORANGE),
            ("GPU 1", gpu1.elapsed_s.to_numpy(float) / 3600, gpu1.gpu_memory_used_mib.to_numpy(float) / 1024, BLUE),
        ],
        "GiB",
    )
    draw_chart(
        draw,
        boxes[1],
        "B. Cgroup memory composition",
        [
            ("total", time.elapsed_s.to_numpy(float) / 3600, time.cgroup_current_bytes.to_numpy(float) / 2**30, ORANGE),
            ("anonymous", time.elapsed_s.to_numpy(float) / 3600, time.cgroup_anon_bytes.to_numpy(float) / 2**30, PURPLE),
            ("file/cache", time.elapsed_s.to_numpy(float) / 3600, time.cgroup_file_bytes.to_numpy(float) / 2**30, BLUE),
        ],
        "GiB",
        reference=240.0,
    )
    draw_chart(
        draw,
        boxes[2],
        "C. Cgroup growth from each run's own start",
        resource_growth_series(),
        "GiB added",
    )
    draw_chart(
        draw,
        boxes[3],
        "D. Memory-limit event counters",
        [
            ("max", time.elapsed_s.to_numpy(float) / 3600, time.event_max.to_numpy(float), RED),
            ("oom", time.elapsed_s.to_numpy(float) / 3600, time.event_oom.to_numpy(float), ORANGE),
            ("oom_kill", time.elapsed_s.to_numpy(float) / 3600, time.event_oom_kill.to_numpy(float), PURPLE),
        ],
        "cumulative count",
        reference=0.0,
    )
    image.save(OUT / "V2_RESOURCES_G51.png", optimize=True)


def summarize_runs(runs: dict[str, tuple[pd.DataFrame, str]], latest: int) -> dict[str, object]:
    common = min(latest, *(int(frame.step.max()) for frame, _ in runs.values()))
    result: dict[str, object] = {"common_window_end": common, "runs": {}}
    for label, (frame, _) in runs.items():
        own = frame.copy()
        overlap = frame[frame.step <= common].copy()
        result["runs"][label] = {
            "latest_available_step": int(own.step.max()),
            "latest_success_pct": float(own.iloc[-1].success_pct),
            "own_window_success_mean_pct": float(own.success_pct.mean()),
            "common_g1_to_g51_success_mean_pct": float(overlap.success_pct.mean()),
            "common_latest5_success_pct": float(overlap.tail(5).success_pct.mean()),
            "common_latest10_success_pct": float(overlap.tail(10).success_pct.mean()),
            "common_approx_kl_mean": float(overlap.approx_kl.mean()),
            "common_clip_pct_mean": float(overlap.clip_pct.mean()),
            "common_grad_norm_mean": float(overlap.grad_norm.mean()),
            "common_step_minutes_mean": float(overlap.step_min.mean()),
        }
    baseline = result["runs"]["Original GRPO"]
    result["deltas_vs_grpo_common_window"] = {}
    for label in ("DVAC v1 global-z", "DVAC v2 R-only"):
        row = result["runs"][label]
        result["deltas_vs_grpo_common_window"][label] = {
            "g1_to_g51_mean_success_pp": row["common_g1_to_g51_success_mean_pct"] - baseline["common_g1_to_g51_success_mean_pct"],
            "latest5_success_pp": row["common_latest5_success_pct"] - baseline["common_latest5_success_pct"],
            "latest10_success_pp": row["common_latest10_success_pct"] - baseline["common_latest10_success_pct"],
        }
    return result


def main() -> None:
    base = load_base()
    OUT.mkdir(parents=True, exist_ok=True)
    runs, _, _, current_raw = load_runs(base)
    latest = int(current_raw.global_step.max())
    if latest != 51:
        raise RuntimeError(f"closeout contract expected g51, parsed g{latest}")

    rank_paths = sorted((RAW / "run/dvac_train").glob("actor_rank*/runner_step_metrics.csv"))
    npz_paths = sorted((RAW / "run/dvac_train").glob("actor_rank*/rollout_step0050.npz"))
    rank, dvac = base.aggregate_dvac_steps(rank_paths)
    horizon, horizon_summary = base.latest_horizon(npz_paths)
    resources = pd.read_csv(RAW / "runtime/resource_monitor/resources.csv")

    plot_success(runs, latest)
    plot_optimization(runs)
    plot_method(dvac, horizon, latest)
    plot_resources(resources)

    rows: list[dict[str, object]] = []
    for label, (frame, _) in runs.items():
        success = frame.success_pct.to_numpy(float)
        for index, row in frame.iterrows():
            rows.append(
                {
                    "run": label,
                    "step": int(row.step),
                    "success_pct": float(row.success_pct),
                    "success_roll5_pct": float(rolling(success, 5)[index]),
                    "approx_kl": float(row.approx_kl),
                    "clip_pct": float(row.clip_pct),
                    "grad_norm": float(row.grad_norm),
                    "step_min": float(row.step_min),
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "THREE_RUN_METRICS_G51.csv", index=False)
    rank.to_csv(OUT / "V2_DVAC_RANK_METRICS_G51.csv", index=False)
    dvac.to_csv(OUT / "V2_DVAC_STEP_METRICS_G51.csv", index=False)
    horizon.to_csv(OUT / "V2_LATEST_HORIZON_G51.csv", index=False)

    time = resources.drop_duplicates("elapsed_s", keep="last").sort_values("elapsed_s")
    gpu0 = resources[resources.gpu_index == 0]
    gpu1 = resources[resources.gpu_index == 1]
    event_rows = time[time.event_max > 0]
    summary = {
        "snapshot": {
            "stop_completed_time": "2026-08-22T21:42:04+08:00",
            "latest_complete_global_step": latest,
            "termination": "user-authorized stop after complete g51; SIGINT then exact-driver SIGTERM",
            "controllers": "wrapper/driver/observer exited",
            "checkpoints": ["global_step_10", "global_step_20", "global_step_30", "global_step_40", "global_step_50"],
            "run_size_reported": "49 GiB",
            "runtime_size_reported": "110 MiB",
            "resource_csv_through": str(time.iloc[-1].timestamp),
        },
        "three_run_training": summarize_runs(runs, latest),
        "v2_method_g2_to_g51": {
            "weight_mean": float(dvac[dvac.warmup == 0].weight_mean.mean()),
            "weight_p05_mean": float(dvac[dvac.warmup == 0].weight_p05.mean()),
            "weight_median_mean": float(dvac[dvac.warmup == 0].weight_p50.mean()),
            "weight_p95_mean": float(dvac[dvac.warmup == 0].weight_p95.mean()),
            "downweighted_fraction_mean": float(dvac[dvac.warmup == 0].weight_below_one_fraction.mean()),
            "upweighted_fraction_mean": float(dvac[dvac.warmup == 0].weight_above_one_fraction.mean()),
            "lower_bound_fraction_mean": float(dvac[dvac.warmup == 0].z_low_clip_fraction.mean()),
            "upper_bound_fraction_mean": float(dvac[dvac.warmup == 0].z_high_clip_fraction.mean()),
            "positive_adv_weight_mean": float(dvac[dvac.warmup == 0].positive_adv_weight_mean.mean()),
            "negative_adv_weight_mean": float(dvac[dvac.warmup == 0].negative_adv_weight_mean.mean()),
            "latest_horizon": horizon_summary,
            "latest_weight_geometry": latest_weight_geometry(npz_paths),
        },
        "v2_resources": {
            "gpu0_peak_gib": float(gpu0.gpu_memory_used_mib.max() / 1024),
            "gpu1_peak_gib": float(gpu1.gpu_memory_used_mib.max() / 1024),
            "cgroup_peak_gib": float(time.cgroup_current_bytes.max() / 2**30),
            "cgroup_latest_gib": float(time.iloc[-1].cgroup_current_bytes / 2**30),
            "cgroup_limit_gib": 240.0,
            "anon_peak_gib": float(time.cgroup_anon_bytes.max() / 2**30),
            "file_peak_gib": float(time.cgroup_file_bytes.max() / 2**30),
            "event_max_latest": int(time.event_max.max()),
            "event_max_first_nonzero_timestamp": None if event_rows.empty else str(event_rows.iloc[0].timestamp),
            "event_oom_latest": int(time.event_oom.max()),
            "event_oom_kill_latest": int(time.event_oom_kill.max()),
            "host_ram_available_min_gib": float(time.host_mem_available_kib.min() / 2**20),
            "disk_available_min_gib": float(time.disk_available_kib.min() / 2**20),
        },
    }
    (OUT / "SUMMARY_G51.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
