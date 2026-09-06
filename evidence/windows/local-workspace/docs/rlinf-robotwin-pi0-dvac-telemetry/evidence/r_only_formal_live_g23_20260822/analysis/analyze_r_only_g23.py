from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE.parent
EXTRACTED = SNAPSHOT / "extracted"
WORKSPACE = Path(__file__).resolve().parents[5]
BASELINE_JSON = WORKSPACE / "audits" / "20260717-084926-grpo-current" / "analysis.json"

FONT_REGULAR = Path(r"C:\Windows\Fonts\segoeui.ttf")
FONT_SEMIBOLD = Path(r"C:\Windows\Fonts\seguisb.ttf")
INK = "#14213d"
MUTED = "#5e6c84"
GRID = "#d9e1ea"
BLUE = "#2563eb"
ORANGE = "#ea580c"
GREEN = "#15803d"
PURPLE = "#7c3aed"
GOLD = "#d97706"
WHITE = "#ffffff"


def font(size: int, semibold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_SEMIBOLD if semibold else FONT_REGULAR), size)


def one(pattern: str) -> Path:
    matches = sorted(EXTRACTED.rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern!r}, found {len(matches)}: {matches}")
    return matches[0]


def parse_duration(text: str) -> float:
    parts = [int(part) for part in text.split(":")]
    if len(parts) == 2:
        return float(parts[0] * 60 + parts[1])
    if len(parts) == 3:
        return float(parts[0] * 3600 + parts[1] * 60 + parts[2])
    raise ValueError(text)


def parse_metrics(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="replace")
    starts = list(
        re.finditer(r"Global Step:\s*(\d+)\s*/\s*(\d+).*?│\s*([0-9.]+)%", text)
    )
    metric_pattern = re.compile(
        r"\s*([A-Za-z][A-Za-z0-9_/]*)=(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*"
    )
    header_pattern = re.compile(
        r"Elapsed:\s*([0-9:]+)\s*│\s*ETA:\s*([0-9:]+)\s*│\s*Step Time:\s*([0-9.]+)s"
    )
    rows: list[dict[str, float | int | str]] = []
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        block = text[start.start() : end]
        header = header_pattern.search(block)
        if not header:
            continue
        elapsed, eta, mean_step_time = header.groups()
        row: dict[str, float | int | str] = {
            "global_step": int(start.group(1)),
            "target_step": int(start.group(2)),
            "progress_pct": float(start.group(3)),
            "elapsed_text": elapsed,
            "elapsed_s": parse_duration(elapsed),
            "eta_text": eta,
            "eta_s": parse_duration(eta),
            "mean_step_time_s": float(mean_step_time),
        }
        for segment in block.replace("\n", "│").split("│"):
            match = metric_pattern.fullmatch(segment)
            if match:
                row[match.group(1)] = float(match.group(2))
        if "step" in row:
            row["step_time_s"] = float(row["step"])
        if "success_once" in row:
            row["success_pct"] = 100.0 * float(row["success_once"])
        rows.append(row)
    if not rows:
        raise RuntimeError(f"No complete Global Step records parsed from {path}")
    frame = pd.DataFrame(rows).drop_duplicates("global_step", keep="last")
    return frame.sort_values("global_step").reset_index(drop=True)


def weighted_mean(group: pd.DataFrame, column: str) -> float:
    valid = group[[column, "valid_queries_local"]].dropna()
    if valid.empty:
        return float("nan")
    weights = valid.valid_queries_local.to_numpy(float)
    values = valid[column].to_numpy(float)
    if weights.sum() == 0:
        return float(np.mean(values))
    return float(np.average(values, weights=weights))


def aggregate_dvac_steps(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rank = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    numeric = [column for column in rank.columns if column not in {"mode", "signal_mode"}]
    rank[numeric] = rank[numeric].apply(pd.to_numeric, errors="coerce")
    rank["global_step"] = rank.runner_step + 1
    rows: list[dict[str, float | int | str]] = []
    weighted_columns = (
        "weight_mean",
        "weight_below_one_fraction",
        "weight_above_one_fraction",
        "z_low_clip_fraction",
        "z_high_clip_fraction",
    )
    rank_mean_columns = (
        "weight_p05",
        "weight_p50",
        "weight_p95",
        "residual_p05",
        "residual_p50",
        "residual_p95",
        "positive_adv_weight_mean",
        "negative_adv_weight_mean",
        "current_mean",
        "current_std",
        "history_mean",
        "history_std",
        "position_center_median",
        "position_scale_median",
        "actor_grad_norm",
        "actor_clip_fraction",
        "actor_approx_kl",
        "actor_ratio",
        "actor_policy_loss",
    )
    for step, group in rank.groupby("global_step", sort=True):
        row: dict[str, float | int | str] = {
            "global_step": int(step),
            "runner_step": int(step - 1),
            "signal_mode": str(group.signal_mode.iloc[0]),
            "warmup": int(group.warmup.max()),
            "rank_count": int(group.actor_rank.nunique()),
            "valid_queries": int(group.valid_queries_local.sum()),
            "history_steps": int(group.history_steps.max()),
            # The two ranks receive the same globally gathered history; do not sum it.
            "history_queries": int(group.history_queries.max()),
            "weight_min": float(group.weight_min.min()),
            "weight_max": float(group.weight_max.max()),
        }
        for column in weighted_columns:
            row[column] = weighted_mean(group, column)
        for column in rank_mean_columns:
            row[column] = float(group[column].mean())
        rows.append(row)
    return rank, pd.DataFrame(rows)


def latest_horizon(npz_paths: list[Path]) -> tuple[pd.DataFrame, dict[str, float]]:
    query_arrays: dict[str, list[np.ndarray]] = {
        key: [] for key in ("log_v_l3", "weights", "residual_z", "advantages")
    }
    centers: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    for path in npz_paths:
        with np.load(path) as data:
            mask = data["loss_mask"].reshape(-1).astype(bool)
            log_v = np.log(data["v_l3"].reshape(-1, 50) + 1e-12)[mask]
            weights = data["weights"].reshape(-1, 50)[mask]
            residual = data["residual_z"].reshape(-1, 50)[mask]
            advantage = data["advantages"].reshape(-1)[mask]
            query_arrays["log_v_l3"].append(log_v)
            query_arrays["weights"].append(weights)
            query_arrays["residual_z"].append(residual)
            query_arrays["advantages"].append(advantage)
            centers.append(data["position_center_log_v"].astype(np.float64))
            scales.append(data["position_scale_log_v"].astype(np.float64))
    values = {key: np.concatenate(items, axis=0) for key, items in query_arrays.items()}
    center = np.mean(np.stack(centers), axis=0)
    scale = np.mean(np.stack(scales), axis=0)
    rows: list[dict[str, float | int]] = []
    for h in range(50):
        w = values["weights"][:, h]
        r = values["residual_z"][:, h]
        lv = values["log_v_l3"][:, h]
        rows.append(
            {
                "h": h,
                "position_center_log_v": center[h],
                "position_scale_log_v": scale[h],
                "current_log_v_median": float(np.median(lv)),
                "residual_median": float(np.median(r)),
                "weight_mean": float(np.mean(w)),
                "weight_p05": float(np.quantile(w, 0.05)),
                "weight_p50": float(np.quantile(w, 0.50)),
                "weight_p95": float(np.quantile(w, 0.95)),
                "below_one_fraction": float(np.mean(w < 1.0)),
                "above_one_fraction": float(np.mean(w > 1.0)),
                "at_min_fraction": float(np.mean(w <= 0.500001)),
                "at_max_fraction": float(np.mean(w >= 1.199999)),
            }
        )
    horizon = pd.DataFrame(rows)
    front = slice(0, 25)
    back = slice(25, 50)
    valid_weights = values["weights"].reshape(-1)
    advantages = values["advantages"]
    query_weight = values["weights"].mean(axis=1)
    summary = {
        "valid_queries": int(values["weights"].shape[0]),
        "valid_weight_points": int(valid_weights.size),
        "weight_mean": float(np.mean(valid_weights)),
        "weight_p05": float(np.quantile(valid_weights, 0.05)),
        "weight_p50": float(np.quantile(valid_weights, 0.50)),
        "weight_p95": float(np.quantile(valid_weights, 0.95)),
        "weight_below_one_fraction": float(np.mean(valid_weights < 1.0)),
        "weight_above_one_fraction": float(np.mean(valid_weights > 1.0)),
        "weight_at_min_fraction": float(np.mean(valid_weights <= 0.500001)),
        "weight_at_max_fraction": float(np.mean(valid_weights >= 1.199999)),
        "positive_adv_query_weight_mean": float(np.mean(query_weight[advantages > 0])),
        "negative_adv_query_weight_mean": float(np.mean(query_weight[advantages < 0])),
        "raw_log_v_back_minus_front": float(
            np.mean(values["log_v_l3"][:, back]) - np.mean(values["log_v_l3"][:, front])
        ),
        "position_center_back_minus_front": float(np.mean(center[back]) - np.mean(center[front])),
        "residual_back_minus_front": float(
            np.mean(values["residual_z"][:, back]) - np.mean(values["residual_z"][:, front])
        ),
        "weight_back_minus_front": float(
            np.mean(values["weights"][:, back]) - np.mean(values["weights"][:, front])
        ),
        "weight_h_spearman": float(horizon.h.rank().corr(horizon.weight_mean.rank())),
        "center_rank_max_abs_difference": float(np.max(np.abs(centers[0] - centers[1]))),
        "scale_rank_max_abs_difference": float(np.max(np.abs(scales[0] - scales[1]))),
    }
    return horizon, summary


def draw_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    xs: np.ndarray,
    series: list[tuple[str, np.ndarray, str]],
    y_label: str,
    reference: float | None = None,
) -> None:
    left, top, right, bottom = box
    draw.text((left, top), title, fill=INK, font=font(25, True))
    plot_left, plot_right = left + 88, right - 24
    plot_top, plot_bottom = top + 72, bottom - 45
    finite_values = [
        float(value)
        for _, values, _ in series
        for value in values
        if np.isfinite(value)
    ]
    if reference is not None:
        finite_values.append(float(reference))
    low, high = min(finite_values), max(finite_values)
    span = high - low
    pad = 0.12 * span if span else max(abs(high) * 0.1, 1.0)
    low, high = low - pad, high + pad
    x_low, x_high = float(np.nanmin(xs)), float(np.nanmax(xs))
    if x_high == x_low:
        x_high += 1.0

    def px(value: float) -> float:
        return plot_left + (value - x_low) / (x_high - x_low) * (plot_right - plot_left)

    def py(value: float) -> float:
        return plot_bottom - (value - low) / (high - low) * (plot_bottom - plot_top)

    for tick in range(5):
        value = low + tick * (high - low) / 4
        y = py(value)
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=1)
        label = f"{value:.3f}" if abs(value) < 2 else f"{value:.1f}"
        width = draw.textlength(label, font=font(15))
        draw.text((plot_left - width - 10, y - 9), label, fill=MUTED, font=font(15))
    for tick in range(5):
        value = x_low + tick * (x_high - x_low) / 4
        label = f"{value:.1f}" if x_high < 10 else f"{value:.0f}"
        x = px(value)
        width = draw.textlength(label, font=font(15))
        draw.text((x - width / 2, plot_bottom + 8), label, fill=MUTED, font=font(15))
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=INK, width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=INK, width=2)
    if reference is not None and low <= reference <= high:
        y = py(reference)
        for x in range(int(plot_left), int(plot_right), 14):
            draw.line((x, y, min(x + 7, plot_right), y), fill=MUTED, width=1)
    legend_x = plot_left
    for label, values, color in series:
        valid = [(float(x), float(y)) for x, y in zip(xs, values) if np.isfinite(y)]
        if len(valid) > 1:
            step = max(1, len(valid) // 1800)
            points = [(px(x), py(y)) for x, y in valid[::step]]
            if points[-1] != (px(valid[-1][0]), py(valid[-1][1])):
                points.append((px(valid[-1][0]), py(valid[-1][1])))
            draw.line(points, fill=color, width=4, joint="curve")
        elif valid:
            x, y = px(valid[0][0]), py(valid[0][1])
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        draw.line((legend_x, plot_top - 24, legend_x + 24, plot_top - 24), fill=color, width=5)
        draw.text((legend_x + 31, plot_top - 35), label, fill=INK, font=font(16))
        legend_x += int(draw.textlength(label, font=font(16))) + 78
    draw.text((left, plot_top - 2), y_label, fill=MUTED, font=font(15))


def canvas(title: str, subtitle: str, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (1440, height), WHITE)
    draw = ImageDraw.Draw(image)
    draw.text((66, 42), title, fill=INK, font=font(37, True))
    draw.text((68, 94), subtitle, fill=MUTED, font=font(20))
    return image, draw


def panel_boxes(rows: int) -> list[tuple[int, int, int, int]]:
    top, row_height, gap = 155, 400, 24
    boxes: list[tuple[int, int, int, int]] = []
    for row in range(rows):
        y = top + row * (row_height + gap)
        boxes.extend(((48, y, 708, y + row_height), (732, y, 1392, y + row_height)))
    return boxes


def plot_training(current: pd.DataFrame, baseline: pd.DataFrame) -> None:
    overlap = int(current.global_step.max())
    merged = current.merge(
        baseline[baseline.step <= overlap], left_on="global_step", right_on="step", suffixes=("_r", "_g")
    )
    x = merged.global_step.to_numpy(float)
    image, draw = canvas(
        f"R-only DVAC through Global Step {overlap} vs historical GRPO",
        "Same global-step axis; success is on-policy training rollout success, not held-out evaluation.",
        1460,
    )
    boxes = panel_boxes(3)
    draw_chart(
        draw,
        boxes[0],
        "Training-rollout success (5-step mean)",
        x,
        [
            ("R-only", 100 * merged.success_once_r.rolling(5, min_periods=1).mean().to_numpy(), ORANGE),
            ("GRPO", 100 * merged.success_once_g.rolling(5, min_periods=1).mean().to_numpy(), BLUE),
        ],
        "%",
    )
    delta5 = 100 * (
        merged.success_once_r.rolling(5, min_periods=5).mean()
        - merged.success_once_g.rolling(5, min_periods=5).mean()
    )
    delta10 = 100 * (
        merged.success_once_r.rolling(10, min_periods=10).mean()
        - merged.success_once_g.rolling(10, min_periods=10).mean()
    )
    draw_chart(draw, boxes[1], "R-only minus GRPO success", x, [("5-step delta", delta5.to_numpy(), GOLD), ("10-step delta", delta10.to_numpy(), PURPLE)], "percentage points", 0)
    for box, key, title, label, scale, reference in (
        (boxes[2], "actor/approx_kl", "Approx KL", "fraction", 1.0, None),
        (boxes[3], "actor/clip_fraction", "PPO joint-query clip fraction", "%", 100.0, None),
        (boxes[4], "actor/grad_norm", "Pre-global-clip gradient norm", "norm", 1.0, 1.0),
        (boxes[5], "step_time_s", "Global-step wall time", "minutes", 1 / 60, None),
    ):
        draw_chart(
            draw,
            box,
            title,
            x,
            [("R-only", scale * merged[f"{key}_r"].to_numpy(float), ORANGE), ("historical GRPO", scale * merged[f"{key}_g"].to_numpy(float), BLUE)],
            label,
            reference,
        )
    image.save(HERE / "TRAINING_VS_GRPO_G23.png", optimize=True)


def plot_dvac(step: pd.DataFrame, horizon: pd.DataFrame) -> None:
    apply = step[step.warmup == 0]
    x = apply.global_step.to_numpy(float)
    image, draw = canvas(
        "R-only DVAC weighting diagnostics",
        "Global Step 1 is warmup (w=1); Steps 2-23 use per-h robust residual mapped to [0.5, 1.2].",
        1035,
    )
    boxes = panel_boxes(2)
    draw_chart(
        draw,
        boxes[0],
        "Per-action gradient weights",
        x,
        [("p05", apply.weight_p05.to_numpy(), BLUE), ("median", apply.weight_p50.to_numpy(), PURPLE), ("mean", apply.weight_mean.to_numpy(), GREEN), ("p95", apply.weight_p95.to_numpy(), ORANGE)],
        "weight",
        1,
    )
    draw_chart(
        draw,
        boxes[1],
        "Weight direction and residual clipping",
        x,
        [
            ("downweighted", 100 * apply.weight_below_one_fraction.to_numpy(), BLUE),
            ("upweighted", 100 * apply.weight_above_one_fraction.to_numpy(), ORANGE),
            ("at -2", 100 * apply.z_low_clip_fraction.to_numpy(), PURPLE),
            ("at +2", 100 * apply.z_high_clip_fraction.to_numpy(), GOLD),
        ],
        "% of valid weight points",
    )
    draw_chart(
        draw,
        boxes[2],
        "Position-calibrated residual signal",
        x,
        [("p05", apply.residual_p05.to_numpy(), BLUE), ("median", apply.residual_p50.to_numpy(), PURPLE), ("p95", apply.residual_p95.to_numpy(), ORANGE)],
        "clipped robust residual",
        0,
    )
    draw_chart(
        draw,
        boxes[3],
        "Latest weights by future-action index",
        horizon.h.to_numpy(float),
        [("p05", horizon.weight_p05.to_numpy(), BLUE), ("mean", horizon.weight_mean.to_numpy(), ORANGE), ("p95", horizon.weight_p95.to_numpy(), GOLD)],
        "weight",
        1,
    )
    image.save(HERE / "DVAC_R_ONLY_DIAGNOSTICS_G23.png", optimize=True)


def plot_resources(resources: pd.DataFrame, baseline_payload: dict) -> dict[str, float | int]:
    resources = resources.sort_values(["elapsed_s", "gpu_index"])
    peak = baseline_payload["peak"]
    baseline_gpu_peak = max(float(peak["peak_gpu0_mb"]), float(peak["peak_gpu1_mb"])) / 1024
    host = resources.drop_duplicates("elapsed_s")
    image, draw = canvas(
        "R-only formal resource telemetry",
        "Observer only records resources; dashed references are peaks from the historical successful GRPO run.",
        1035,
    )
    boxes = panel_boxes(2)
    gpu0, gpu1 = resources[resources.gpu_index == 0], resources[resources.gpu_index == 1]
    draw_chart(
        draw,
        boxes[0],
        "GPU memory",
        gpu0.elapsed_s.to_numpy(float) / 3600,
        [("GPU 0", gpu0.gpu_memory_used_mib.to_numpy(float) / 1024, ORANGE), ("GPU 1", gpu1.gpu_memory_used_mib.to_numpy(float) / 1024, BLUE)],
        "GiB",
        baseline_gpu_peak,
    )
    draw_chart(
        draw,
        boxes[1],
        "GPU utilization",
        gpu0.elapsed_s.to_numpy(float) / 3600,
        [("GPU 0", gpu0.gpu_util_pct.to_numpy(float), ORANGE), ("GPU 1", gpu1.gpu_util_pct.to_numpy(float), BLUE)],
        "%",
    )
    draw_chart(
        draw,
        boxes[2],
        "Cgroup RAM",
        host.elapsed_s.to_numpy(float) / 3600,
        [
            ("current", host.cgroup_current_bytes.to_numpy(float) / 2**30, ORANGE),
            ("anonymous", host.cgroup_anon_bytes.to_numpy(float) / 2**30, PURPLE),
            ("file/cache", host.cgroup_file_bytes.to_numpy(float) / 2**30, BLUE),
        ],
        "GiB",
        float(peak["peak_ram_mb"]) / 1024,
    )
    draw_chart(
        draw,
        boxes[3],
        "Available host resources",
        host.elapsed_s.to_numpy(float) / 3600,
        [("disk", host.disk_available_kib.to_numpy(float) / 2**20, GREEN), ("host RAM", host.host_mem_available_kib.to_numpy(float) / 2**20, GOLD)],
        "GiB",
    )
    image.save(HERE / "RESOURCES_G23.png", optimize=True)
    gpu0 = resources[resources.gpu_index == 0]
    gpu1 = resources[resources.gpu_index == 1]
    return {
        "samples_per_gpu": int(len(gpu0)),
        "elapsed_hours": float(resources.elapsed_s.max() / 3600),
        "gpu0_peak_gib": float(gpu0.gpu_memory_used_mib.max() / 1024),
        "gpu1_peak_gib": float(gpu1.gpu_memory_used_mib.max() / 1024),
        "gpu0_latest_gib": float(gpu0.iloc[-1].gpu_memory_used_mib / 1024),
        "gpu1_latest_gib": float(gpu1.iloc[-1].gpu_memory_used_mib / 1024),
        "gpu0_mean_util_pct": float(gpu0.gpu_util_pct.mean()),
        "gpu1_mean_util_pct": float(gpu1.gpu_util_pct.mean()),
        "cgroup_peak_gib": float(host.cgroup_current_bytes.max() / 2**30),
        "cgroup_latest_gib": float(host.iloc[-1].cgroup_current_bytes / 2**30),
        "cgroup_anon_peak_gib": float(host.cgroup_anon_bytes.max() / 2**30),
        "cgroup_file_peak_gib": float(host.cgroup_file_bytes.max() / 2**30),
        "host_ram_available_min_gib": float(host.host_mem_available_kib.min() / 2**20),
        "disk_available_min_gib": float(host.disk_available_kib.min() / 2**20),
        "shm_available_min_gib": float(host.shm_available_kib.min() / 2**20),
        "event_high_max": int(host.event_high.max()),
        "event_max_max": int(host.event_max.max()),
        "event_oom_max": int(host.event_oom.max()),
        "event_oom_kill_max": int(host.event_oom_kill.max()),
        "historical_grpo_gpu_peak_max_gib": baseline_gpu_peak,
        "historical_grpo_cgroup_peak_gib": float(peak["peak_ram_mb"]) / 1024,
    }


def mean_block(frame: pd.DataFrame, key: str, start: int, end: int, step_key: str) -> float:
    block = frame[frame[step_key].between(start, end)]
    return float(block[key].mean())


def main() -> None:
    metrics_path = one("snapshot_metrics_g23.log")
    rank_csvs = sorted(EXTRACTED.rglob("runner_step_metrics.csv"))
    npz_paths = sorted(EXTRACTED.rglob("rollout_step0022.npz"))
    resource_path = one("snapshot_resources_g23.csv")
    if len(rank_csvs) != 2 or len(npz_paths) != 2:
        raise RuntimeError(f"Expected two rank CSVs/NPZs, got {len(rank_csvs)}/{len(npz_paths)}")
    baseline_payload = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    baseline = pd.DataFrame(baseline_payload["metrics"])
    current = parse_metrics(metrics_path)
    rank, dvac = aggregate_dvac_steps(rank_csvs)
    horizon, latest_h = latest_horizon(npz_paths)
    resources = pd.read_csv(resource_path)

    overlap = int(current.global_step.max())
    base_overlap = baseline[baseline.step <= overlap].copy()
    comparison = current.merge(
        base_overlap, left_on="global_step", right_on="step", suffixes=("_r_only", "_grpo")
    )
    comparison["success_delta"] = comparison.success_once_r_only - comparison.success_once_grpo
    comparison["success_roll5_r_only"] = comparison.success_once_r_only.rolling(5, min_periods=1).mean()
    comparison["success_roll5_grpo"] = comparison.success_once_grpo.rolling(5, min_periods=1).mean()
    comparison["success_roll5_delta"] = comparison.success_roll5_r_only - comparison.success_roll5_grpo

    current.to_csv(HERE / "TRAIN_METRICS_G23.csv", index=False)
    rank.to_csv(HERE / "DVAC_RANK_METRICS_G23.csv", index=False)
    dvac.to_csv(HERE / "DVAC_STEP_METRICS_G23.csv", index=False)
    horizon.to_csv(HERE / "LATEST_HORIZON_G23.csv", index=False)
    comparison.to_csv(HERE / "TRAINING_VS_GRPO_G23.csv", index=False)

    plot_training(current, baseline)
    plot_dvac(dvac, horizon)
    resource_summary = plot_resources(resources, baseline_payload)

    apply = dvac[dvac.warmup == 0]
    latest = current.iloc[-1]
    summary = {
        "snapshot": {
            "latest_complete_global_step": overlap,
            "target_step": int(latest.target_step),
            "progress_pct": float(latest.progress_pct),
            "elapsed_text": str(latest.elapsed_text),
            "eta_text": str(latest.eta_text),
            "cumulative_training_trajectories": int(current.num_trajectories.sum()),
            "current_source": "R-only per-h robust residual DVAC formal",
            "comparison_source": "historical successful GRPO, same global-step axis",
        },
        "training_rollout_success": {
            "current_mean_g1_to_latest": float(current.success_once.mean()),
            "grpo_mean_same_steps": float(base_overlap.success_once.mean()),
            "mean_delta_percentage_points": float(100 * (current.success_once.mean() - base_overlap.success_once.mean())),
            "current_latest": float(latest.success_once),
            "grpo_same_step": float(base_overlap.iloc[-1].success_once),
            "latest_delta_percentage_points": float(100 * (latest.success_once - base_overlap.iloc[-1].success_once)),
            "current_recent5": mean_block(current, "success_once", max(1, overlap - 4), overlap, "global_step"),
            "grpo_recent5": mean_block(baseline, "success_once", max(1, overlap - 4), overlap, "step"),
            "current_recent10": mean_block(current, "success_once", max(1, overlap - 9), overlap, "global_step"),
            "grpo_recent10": mean_block(baseline, "success_once", max(1, overlap - 9), overlap, "step"),
            "metric_boundary": "on-policy training rollout success; not held-out evaluation",
        },
        "optimization_means_g1_to_latest": {
            "r_only_approx_kl": float(current["actor/approx_kl"].mean()),
            "grpo_approx_kl": float(base_overlap["actor/approx_kl"].mean()),
            "r_only_clip_fraction": float(current["actor/clip_fraction"].mean()),
            "grpo_clip_fraction": float(base_overlap["actor/clip_fraction"].mean()),
            "r_only_preclip_grad_norm": float(current["actor/grad_norm"].mean()),
            "grpo_preclip_grad_norm": float(base_overlap["actor/grad_norm"].mean()),
            "r_only_step_time_min": float(current.step_time_s.mean() / 60),
            "grpo_step_time_min": float(base_overlap.step_time_s.mean() / 60),
        },
        "dvac_apply_steps_g2_to_latest": {
            "steps": int(len(apply)),
            "rank_weighted_mean_weight": float(apply.weight_mean.mean()),
            "rank_mean_p05": float(apply.weight_p05.mean()),
            "rank_mean_median": float(apply.weight_p50.mean()),
            "rank_mean_p95": float(apply.weight_p95.mean()),
            "downweighted_fraction": float(apply.weight_below_one_fraction.mean()),
            "upweighted_fraction": float(apply.weight_above_one_fraction.mean()),
            "at_lower_residual_clip_fraction": float(apply.z_low_clip_fraction.mean()),
            "at_upper_residual_clip_fraction": float(apply.z_high_clip_fraction.mean()),
            "positive_adv_weight_mean": float(apply.positive_adv_weight_mean.mean()),
            "negative_adv_weight_mean": float(apply.negative_adv_weight_mean.mean()),
            "latest_npz_exact": latest_h,
        },
        "resources": resource_summary,
        "notes": [
            "The historical GRPO run is an engineering-aligned historical reference, not a paired same-seed control arm.",
            "Both methods have pre-clip gradient norms far above global clip_grad=1; DVAC primarily changes relative per-h gradient composition.",
            "R-only removes the recent per-h position baseline before mapping residuals to [0.5,1.2].",
        ],
    }
    (HERE / "ANALYSIS_SUMMARY_G23.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    inventory = [
        {"relative_path": str(path.relative_to(EXTRACTED)), "bytes": path.stat().st_size}
        for path in sorted(EXTRACTED.rglob("*"))
        if path.is_file()
    ]
    with (HERE / "SNAPSHOT_FILE_INVENTORY.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=("relative_path", "bytes"))
        writer.writeheader()
        writer.writerows(inventory)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
