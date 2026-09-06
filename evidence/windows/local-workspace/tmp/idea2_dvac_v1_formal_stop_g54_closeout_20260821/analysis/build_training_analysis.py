from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE.parent
METRICS_LOG = SNAPSHOT / "run" / "metrics.log"
BASELINE_JSON = Path(
    r"C:\Users\86136\Documents\rl\audits\20260717-084926-grpo-current\analysis.json"
)
LAUNCH_TIME_FILE = SNAPSHOT / "runtime" / "launch_started_at.txt"


def parse_duration(text: str) -> float:
    parts = [int(part) for part in text.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Unsupported duration: {text}")


def parse_metrics(path: Path) -> list[dict[str, float | int | str]]:
    text = path.read_text(encoding="utf-8")
    starts = list(
        re.finditer(
            r"Global Step:\s*(\d+)\s*/\s*(\d+).*?│\s*([0-9.]+)%",
            text,
        )
    )
    rows: list[dict[str, float | int | str]] = []
    metric_pattern = re.compile(
        r"\s*([A-Za-z][A-Za-z0-9_/]*)=(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*"
    )
    header_pattern = re.compile(
        r"Elapsed:\s*([0-9:]+)\s*│\s*ETA:\s*([0-9:]+)\s*│\s*Step Time:\s*([0-9.]+)s"
    )
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        block = text[start.start() : end]
        header = header_pattern.search(block)
        if not header:
            continue
        elapsed_text, eta_text, mean_step_time = header.groups()
        row: dict[str, float | int | str] = {
            "global_step": int(start.group(1)),
            "target_step": int(start.group(2)),
            "progress_pct": float(start.group(3)),
            "runner_step_index_zero_based": int(start.group(1)) - 1,
            "elapsed_text": elapsed_text,
            "elapsed_s": parse_duration(elapsed_text),
            "elapsed_h": parse_duration(elapsed_text) / 3600.0,
            "eta_text": eta_text,
            "eta_s": parse_duration(eta_text),
            "eta_h": parse_duration(eta_text) / 3600.0,
            "mean_step_time_s": float(mean_step_time),
        }
        for segment in block.replace("\n", "│").split("│"):
            match = metric_pattern.fullmatch(segment)
            if match:
                row[match.group(1)] = float(match.group(2))
        if "success_once" in row and "num_trajectories" in row:
            row["success_trajectories"] = int(
                round(float(row["success_once"]) * float(row["num_trajectories"]))
            )
        if "step" in row:
            row["step_time_s"] = float(row["step"])
            row["step_time_min"] = float(row["step"]) / 60.0
        if "success_once" in row:
            row["success_pct"] = float(row["success_once"]) * 100.0
        if "actor/dvac_current_mean" in row and "actor/dvac_history_mean" in row:
            row["actor/dvac_current_minus_history"] = (
                float(row["actor/dvac_current_mean"])
                - float(row["actor/dvac_history_mean"])
            )
        rows.append(row)

    cumulative_trajectories = 0
    cumulative_successes = 0
    for i, row in enumerate(rows):
        cumulative_trajectories += int(float(row.get("num_trajectories", 0)))
        cumulative_successes += int(row.get("success_trajectories", 0))
        row["cumulative_trajectories"] = cumulative_trajectories
        row["cumulative_successes"] = cumulative_successes
        for window in (5, 10):
            window_rows = rows[max(0, i - window + 1) : i + 1]
            row[f"success_once_rolling{window}"] = mean(
                float(item["success_once"]) for item in window_rows
            )
            row[f"step_time_s_rolling{window}"] = mean(
                float(item["step_time_s"]) for item in window_rows
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], preferred: list[str]) -> None:
    keys = {key for row in rows for key in row}
    columns = [key for key in preferred if key in keys]
    columns.extend(sorted(keys - set(columns)))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


FONT_REGULAR = Path(r"C:\Windows\Fonts\segoeui.ttf")
FONT_SEMIBOLD = Path(r"C:\Windows\Fonts\seguisb.ttf")


def font(size: int, semibold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(
        str(FONT_SEMIBOLD if semibold else FONT_REGULAR), size=size
    )


INK = "#14213D"
MUTED = "#5E6C84"
GRID = "#D9E1EA"
BLUE = "#1F6FEB"
ORANGE = "#D94801"
TEAL = "#16857B"
PURPLE = "#7A3E9D"
GOLD = "#AD6B00"
RED = "#B42318"
LIGHT_BLUE = "#93B4E8"
LIGHT_ORANGE = "#F1A17B"
WHITE = "#FFFFFF"


def finite_pairs(xs: list[float], ys: list[float | None]) -> list[tuple[float, float]]:
    return [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if y is not None and math.isfinite(float(y))
    ]


def draw_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    xs: list[float],
    series: list[dict[str, object]],
    y_label: str,
    annotation: str = "",
    y_limits: tuple[float, float] | None = None,
    x_limits: tuple[float, float] | None = None,
    reference: float | None = None,
) -> None:
    left, top, right, bottom = box
    draw.text((left, top), title, fill=INK, font=font(27, True))
    if annotation:
        ann_font = font(19)
        ann_width = draw.textlength(annotation, font=ann_font)
        draw.text((right - ann_width, top + 5), annotation, fill=MUTED, font=ann_font)

    plot_top = top + 52
    plot_bottom = bottom - 46
    plot_left = left + 95
    plot_right = right - 28
    all_pairs = [
        pair
        for item in series
        for pair in finite_pairs(xs, item["values"])  # type: ignore[arg-type]
    ]
    if y_limits is None:
        values = [pair[1] for pair in all_pairs]
        low, high = min(values), max(values)
        span = high - low
        pad = span * 0.12 if span else max(abs(high) * 0.1, 1.0)
        low, high = low - pad, high + pad
    else:
        low, high = y_limits
    if x_limits is None:
        x_low, x_high = min(xs), max(xs)
    else:
        x_low, x_high = x_limits
    if x_high == x_low:
        x_high += 1

    def px(x: float) -> float:
        return plot_left + (x - x_low) / (x_high - x_low) * (plot_right - plot_left)

    def py(y: float) -> float:
        return plot_bottom - (y - low) / (high - low) * (plot_bottom - plot_top)

    for tick in range(5):
        value = low + (high - low) * tick / 4
        y = py(value)
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=2)
        label = f"{value:.3f}" if abs(value) < 2 else f"{value:.1f}"
        label_width = draw.textlength(label, font=font(17))
        draw.text((plot_left - label_width - 12, y - 11), label, fill=MUTED, font=font(17))
    if reference is not None and low <= reference <= high:
        y = py(reference)
        for x in range(int(plot_left), int(plot_right), 16):
            draw.line((x, y, min(x + 8, plot_right), y), fill=MUTED, width=2)

    x_tick_count = 5
    for tick in range(x_tick_count):
        value = x_low + (x_high - x_low) * tick / (x_tick_count - 1)
        x = px(value)
        label = str(int(round(value)))
        label_width = draw.textlength(label, font=font(17))
        draw.text((x - label_width / 2, plot_bottom + 10), label, fill=MUTED, font=font(17))

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=INK, width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=INK, width=2)
    draw.text((left, plot_top - 1), y_label, fill=MUTED, font=font(17))

    legend_x = plot_left
    legend_y = plot_top + 5
    for item in series:
        pairs = finite_pairs(xs, item["values"])  # type: ignore[arg-type]
        if not pairs:
            continue
        color = str(item["color"])
        width = int(item.get("width", 4))
        points = [(px(x), py(y)) for x, y in pairs]
        if len(points) > 1:
            draw.line(points, fill=color, width=width, joint="curve")
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        label = str(item["label"])
        draw.line((legend_x, legend_y + 11, legend_x + 28, legend_y + 11), fill=color, width=5)
        draw.text((legend_x + 37, legend_y), label, fill=INK, font=font(18))
        legend_x += int(draw.textlength(label, font=font(18))) + 92


def rolling(values: list[float], window: int) -> list[float]:
    return [mean(values[max(0, i - window + 1) : i + 1]) for i in range(len(values))]


def save_training_plot(rows: list[dict[str, object]], path: Path) -> None:
    image = Image.new("RGB", (1440, 2760), WHITE)
    draw = ImageDraw.Draw(image)
    latest = rows[-1]
    steps = [float(row["global_step"]) for row in rows]
    success = [float(row["success_once"]) * 100 for row in rows]
    draw.text((78, 52), "DVAC formal training — snapshot metrics", fill=INK, font=font(40, True))
    subtitle = (
        f"Latest complete Global Step {int(latest['global_step'])}/100  |  "
        f"rollout success {float(latest['success_once'])*100:.2f}%  |  "
        f"logged ETA {latest['eta_text']}"
    )
    draw.text((80, 110), subtitle, fill=MUTED, font=font(22))
    draw.text(
        (80, 146),
        "success_once/return are training-rollout metrics (256 trajectories per step), not held-out evaluation.",
        fill=MUTED,
        font=font(19),
    )

    chart_left, chart_right = 72, 1370
    top = 215
    height = 400
    gap = 20
    charts = [
        (
            "Rollout success",
            [
                {"label": "per-step success", "values": success, "color": LIGHT_BLUE, "width": 3},
                {"label": "5-step mean", "values": rolling(success, 5), "color": BLUE, "width": 6},
            ],
            "%",
            f"latest {success[-1]:.2f}%  |  best {max(success):.2f}% @ g{steps[success.index(max(success))]:.0f}",
            (70.0, 95.0),
            None,
        ),
        (
            "Step duration",
            [
                {
                    "label": "iteration step",
                    "values": [float(row["step_time_min"]) for row in rows],
                    "color": ORANGE,
                    "width": 5,
                },
                {
                    "label": "running mean",
                    "values": [float(row["mean_step_time_s"]) / 60 for row in rows],
                    "color": GOLD,
                    "width": 4,
                },
            ],
            "min",
            f"latest {float(latest['step_time_min']):.2f} min  |  running mean {float(latest['mean_step_time_s'])/60:.2f} min",
            None,
            None,
        ),
        (
            "PPO movement",
            [
                {"label": "approx KL", "values": [float(row["actor/approx_kl"]) for row in rows], "color": PURPLE, "width": 5},
                {"label": "clip fraction", "values": [float(row["actor/clip_fraction"]) for row in rows], "color": TEAL, "width": 5},
            ],
            "fraction",
            f"latest KL {float(latest['actor/approx_kl']):.3f}  |  clip {float(latest['actor/clip_fraction']):.3f}",
            (0.0, 0.19),
            None,
        ),
        (
            "Gradient and signed policy loss",
            [
                {"label": "grad norm", "values": [float(row["actor/grad_norm"]) for row in rows], "color": RED, "width": 5},
                {"label": "policy loss ×1000", "values": [float(row["actor/policy_loss"]) * 1000 for row in rows], "color": BLUE, "width": 4},
            ],
            "mixed scale",
            f"latest grad {float(latest['actor/grad_norm']):.2f}  |  policy loss {float(latest['actor/policy_loss']):+.4f}",
            None,
            0.0,
        ),
        (
            "Policy ratio",
            [
                {"label": "mean ratio", "values": [float(row["actor/ratio"]) for row in rows], "color": TEAL, "width": 5},
            ],
            "ratio",
            f"latest {float(latest['actor/ratio']):.3f}  |  observed {min(float(r['actor/ratio']) for r in rows):.3f}–{max(float(r['actor/ratio']) for r in rows):.3f}",
            (0.965, 1.015),
            1.0,
        ),
        (
            "DVAC log-variance statistics",
            [
                {"label": "current mean", "values": [float(row["actor/dvac_current_mean"]) for row in rows], "color": ORANGE, "width": 5},
                {"label": "recent-5 history mean", "values": [float(row["actor/dvac_history_mean"]) if int(row["global_step"]) > 1 else None for row in rows], "color": BLUE, "width": 5},
            ],
            "log V",
            f"latest current/history {float(latest['actor/dvac_current_mean']):.3f}/{float(latest['actor/dvac_history_mean']):.3f}  |  rank-local mean w {float(latest['actor/dvac_weight_mean_rank_local']):.3f}",
            (-4.48, -4.12),
            None,
        ),
    ]
    for index, (title, series, label, annotation, limits, reference) in enumerate(charts):
        chart_top = top + index * (height + gap)
        draw_chart(
            draw,
            (chart_left, chart_top, chart_right, chart_top + height),
            title,
            steps,
            series,
            label,
            annotation,
            y_limits=limits,
            x_limits=(1, max(steps)),
            reference=reference,
        )
    image.save(path, optimize=True)


def build_comparison(
    current: list[dict[str, object]], baseline: list[dict[str, object]]
) -> list[dict[str, object]]:
    current_by_step = {int(row["global_step"]): row for row in current}
    baseline_by_step = {int(row["step"]): row for row in baseline}
    metrics = [
        ("success_once", "success_once"),
        ("approx_kl", "actor/approx_kl"),
        ("clip_fraction", "actor/clip_fraction"),
        ("grad_norm", "actor/grad_norm"),
        ("step_time_s", "step_time_s"),
        ("mean_step_time_s", "mean_step_time_s"),
    ]
    rows: list[dict[str, object]] = []
    for step in sorted(baseline_by_step):
        base = baseline_by_step[step]
        cur = current_by_step.get(step)
        row: dict[str, object] = {
            "global_step": step,
            "aligned_runner_index_zero_based": step - 1,
            "baseline_available": 1,
            "current_available": int(cur is not None),
        }
        for short, source in metrics:
            b_value = float(base[source])
            row[f"baseline_{short}"] = b_value
            if cur is not None:
                current_source = source
                if source == "mean_step_time_s":
                    current_source = "mean_step_time_s"
                c_value = float(cur[current_source])
                row[f"current_{short}"] = c_value
                row[f"delta_{short}"] = c_value - b_value
            else:
                row[f"current_{short}"] = ""
                row[f"delta_{short}"] = ""
        rows.append(row)
    return rows


def save_comparison_plot(
    current: list[dict[str, object]], baseline: list[dict[str, object]], path: Path
) -> None:
    overlap = min(int(current[-1]["global_step"]), int(baseline[-1]["step"]))
    cur = [row for row in current if int(row["global_step"]) <= overlap]
    base = [row for row in baseline if int(row["step"]) <= overlap]
    steps = [float(row["global_step"]) for row in cur]
    image = Image.new("RGB", (1440, 2430), WHITE)
    draw = ImageDraw.Draw(image)
    draw.text((78, 52), "DVAC formal vs historical GRPO baseline", fill=INK, font=font(39, True))
    draw.text(
        (80, 110),
        f"Matched Global Steps 1–{overlap}; zero-based runner index = Global Step − 1. Baseline continues to g100.",
        fill=MUTED,
        font=font(21),
    )
    draw.text(
        (80, 145),
        "Same 256 training trajectories/step; success is rollout success, not held-out evaluation.",
        fill=MUTED,
        font=font(19),
    )

    configs = [
        (
            "Rollout success",
            "success_once",
            "success_once",
            "%",
            100.0,
            (70.0, 96.0),
        ),
        ("Approx KL", "actor/approx_kl", "actor/approx_kl", "fraction", 1.0, (0.0, 0.16)),
        ("Clip fraction", "actor/clip_fraction", "actor/clip_fraction", "fraction", 1.0, (0.07, 0.22)),
        ("Pre-clip gradient norm", "actor/grad_norm", "actor/grad_norm", "norm", 1.0, None),
        ("Iteration step time", "step_time_s", "step_time_s", "min", 1.0 / 60.0, None),
    ]
    top, height, gap = 210, 410, 22
    for index, (title, current_key, baseline_key, label, scale, limits) in enumerate(configs):
        current_values = [float(row[current_key]) * scale for row in cur]
        baseline_values = [float(row[baseline_key]) * scale for row in base]
        latest_delta = current_values[-1] - baseline_values[-1]
        mean_delta = mean(current_values) - mean(baseline_values)
        annotation = f"g{overlap} Δ {latest_delta:+.3f}  |  mean Δ {mean_delta:+.3f}"
        draw_chart(
            draw,
            (72, top + index * (height + gap), 1370, top + index * (height + gap) + height),
            title,
            steps,
            [
                {"label": "DVAC formal", "values": current_values, "color": ORANGE, "width": 5},
                {"label": "GRPO baseline", "values": baseline_values, "color": BLUE, "width": 5},
            ],
            label,
            annotation,
            y_limits=limits,
            x_limits=(1, overlap),
        )
    image.save(path, optimize=True)


def summarize(current: list[dict[str, object]], baseline: list[dict[str, object]]) -> dict[str, object]:
    latest = current[-1]
    recent5 = current[-5:]
    recent10 = current[-10:]
    overlap = int(latest["global_step"])
    baseline_overlap = [row for row in baseline if int(row["step"]) <= overlap]
    launch_time = datetime.fromisoformat(LAUNCH_TIME_FILE.read_text(encoding="utf-8").strip())
    snapshot_time = launch_time + timedelta(seconds=float(latest["elapsed_s"]))
    logged_finish = snapshot_time + timedelta(seconds=float(latest["eta_s"]))
    remaining = int(latest["target_step"]) - overlap

    def avg(rows: list[dict[str, object]], key: str) -> float:
        return mean(float(row[key]) for row in rows)

    def metric_comparison(current_key: str, baseline_key: str) -> dict[str, float]:
        return {
            "current_mean_g1_to_latest": avg(current, current_key),
            "baseline_mean_same_steps": avg(baseline_overlap, baseline_key),
            "mean_delta": avg(current, current_key) - avg(baseline_overlap, baseline_key),
            "current_latest": float(latest[current_key]),
            "baseline_same_step": float(baseline_overlap[-1][baseline_key]),
            "latest_delta": float(latest[current_key]) - float(baseline_overlap[-1][baseline_key]),
        }

    best_success_row = max(current, key=lambda row: float(row["success_once"]))
    return {
        "latest_complete_global_step": overlap,
        "runner_step_index_zero_based": overlap - 1,
        "snapshot_training_clock": snapshot_time.isoformat(),
        "latest_logged_eta": latest["eta_text"],
        "logged_projected_finish": logged_finish.isoformat(),
        "recent5_projected_remaining_h": avg(recent5, "step_time_s") * remaining / 3600,
        "recent10_projected_remaining_h": avg(recent10, "step_time_s") * remaining / 3600,
        "cumulative_trajectories": int(latest["cumulative_trajectories"]),
        "cumulative_successes": int(latest["cumulative_successes"]),
        "success": {
            "latest": float(latest["success_once"]),
            "recent5_mean": avg(recent5, "success_once"),
            "recent10_mean": avg(recent10, "success_once"),
            "all_steps_mean": avg(current, "success_once"),
            "best": float(best_success_row["success_once"]),
            "best_step": int(best_success_row["global_step"]),
            "first5_mean": avg(current[:5], "success_once"),
        },
        "step_time_s": {
            "latest_iteration": float(latest["step_time_s"]),
            "logged_running_mean": float(latest["mean_step_time_s"]),
            "recent5_mean": avg(recent5, "step_time_s"),
            "recent10_mean": avg(recent10, "step_time_s"),
        },
        "optimization_recent5": {
            "approx_kl": avg(recent5, "actor/approx_kl"),
            "clip_fraction": avg(recent5, "actor/clip_fraction"),
            "grad_norm": avg(recent5, "actor/grad_norm"),
            "ratio": avg(recent5, "actor/ratio"),
            "policy_loss": avg(recent5, "actor/policy_loss"),
            "policy_loss_abs": avg(recent5, "actor/policy_loss_abs"),
        },
        "optimization_recent10": {
            "approx_kl": avg(recent10, "actor/approx_kl"),
            "clip_fraction": avg(recent10, "actor/clip_fraction"),
            "grad_norm": avg(recent10, "actor/grad_norm"),
            "ratio": avg(recent10, "actor/ratio"),
            "policy_loss": avg(recent10, "actor/policy_loss"),
            "policy_loss_abs": avg(recent10, "actor/policy_loss_abs"),
        },
        "dvac_latest": {
            "current_mean": float(latest["actor/dvac_current_mean"]),
            "current_std": float(latest["actor/dvac_current_std"]),
            "history_mean": float(latest["actor/dvac_history_mean"]),
            "history_std": float(latest["actor/dvac_history_std"]),
            "current_minus_history": float(latest["actor/dvac_current_minus_history"]),
            "warmup": float(latest["actor/dvac_warmup"]),
            "weight_mean_rank_local": float(latest["actor/dvac_weight_mean_rank_local"]),
        },
        "dvac_recent5": {
            "current_mean": avg(recent5, "actor/dvac_current_mean"),
            "current_std": avg(recent5, "actor/dvac_current_std"),
            "history_mean": avg(recent5, "actor/dvac_history_mean"),
            "history_std": avg(recent5, "actor/dvac_history_std"),
            "weight_mean_rank_local": avg(recent5, "actor/dvac_weight_mean_rank_local"),
        },
        "baseline_comparison_g1_to_latest": {
            "success_once": metric_comparison("success_once", "success_once"),
            "approx_kl": metric_comparison("actor/approx_kl", "actor/approx_kl"),
            "clip_fraction": metric_comparison("actor/clip_fraction", "actor/clip_fraction"),
            "grad_norm": metric_comparison("actor/grad_norm", "actor/grad_norm"),
            "step_time_s": metric_comparison("step_time_s", "step_time_s"),
        },
        "baseline_final_success_g100": float(baseline[-1]["success_once"]),
    }


def main() -> None:
    current = parse_metrics(METRICS_LOG)
    baseline_payload = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    baseline = baseline_payload["metrics"]
    if not current or [int(row["global_step"]) for row in current] != list(
        range(1, int(current[-1]["global_step"]) + 1)
    ):
        raise RuntimeError("Current metric tables are missing or non-contiguous")

    preferred = [
        "global_step",
        "target_step",
        "runner_step_index_zero_based",
        "progress_pct",
        "elapsed_text",
        "elapsed_s",
        "elapsed_h",
        "eta_text",
        "eta_s",
        "eta_h",
        "mean_step_time_s",
        "step_time_s",
        "step_time_min",
        "step_time_s_rolling5",
        "step_time_s_rolling10",
        "num_trajectories",
        "success_trajectories",
        "cumulative_trajectories",
        "cumulative_successes",
        "return",
        "success_once",
        "success_pct",
        "success_once_rolling5",
        "success_once_rolling10",
        "reward",
        "actor/approx_kl",
        "actor/clip_fraction",
        "actor/grad_norm",
        "actor/ratio",
        "actor/ratio_abs",
        "actor/policy_loss",
        "actor/policy_loss_abs",
        "actor/total_loss",
        "actor/dvac_current_mean",
        "actor/dvac_current_std",
        "actor/dvac_history_mean",
        "actor/dvac_history_std",
        "actor/dvac_current_minus_history",
        "actor/dvac_warmup",
        "actor/dvac_weight_mean_rank_local",
    ]
    write_csv(HERE / "TRAIN_METRICS.csv", current, preferred)
    save_training_plot(current, HERE / "TRAIN_METRICS.png")

    comparison = build_comparison(current, baseline)
    write_csv(
        HERE / "BASELINE_COMPARISON.csv",
        comparison,
        [
            "global_step",
            "aligned_runner_index_zero_based",
            "baseline_available",
            "current_available",
            "current_success_once",
            "baseline_success_once",
            "delta_success_once",
            "current_approx_kl",
            "baseline_approx_kl",
            "delta_approx_kl",
            "current_clip_fraction",
            "baseline_clip_fraction",
            "delta_clip_fraction",
            "current_grad_norm",
            "baseline_grad_norm",
            "delta_grad_norm",
            "current_step_time_s",
            "baseline_step_time_s",
            "delta_step_time_s",
            "current_mean_step_time_s",
            "baseline_mean_step_time_s",
            "delta_mean_step_time_s",
        ],
    )
    save_comparison_plot(current, baseline, HERE / "BASELINE_COMPARISON.png")

    summary = summarize(current, baseline)
    (HERE / "TRAIN_METRICS_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
