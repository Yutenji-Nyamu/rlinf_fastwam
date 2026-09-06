from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from statistics import mean

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"C:\Users\86136\Documents\rl")
INPUT = ROOT / ".tmp" / "qam_status_20260731_2353"
OUTPUT = Path(
    r"E:\Codex\home\visualizations\2026\07\28\019fa85b-0b0c-72d0-a9ea-a9d88c48e8aa"
)
OUTPUT.mkdir(parents=True, exist_ok=True)


ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
KV_RE = re.compile(rf"([A-Za-z_][A-Za-z0-9_./-]*)=({NUMBER})")
STEP_RE = re.compile(r"Global Step:\s*(\d+)/(\d+)")
ELAPSED_RE = re.compile(r"Elapsed:\s*(\d+):(\d+)")
STEP_TIME_RE = re.compile(rf"Step Time:\s*({NUMBER})s")


def parse_metric_log(path: Path) -> list[dict[str, float]]:
    text = ANSI.sub("", path.read_text(encoding="utf-8", errors="replace"))
    matches = list(STEP_RE.finditer(text))
    rows: list[dict[str, float]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.start() : end]
        row: dict[str, float] = {
            "cycle": float(match.group(1)),
            "step_ceiling": float(match.group(2)),
        }
        elapsed = ELAPSED_RE.search(block)
        if elapsed:
            row["elapsed_seconds"] = float(
                int(elapsed.group(1)) * 60 + int(elapsed.group(2))
            )
        step_time = STEP_TIME_RE.search(block)
        if step_time:
            row["step_time_seconds"] = float(step_time.group(1))
        for key, value in KV_RE.findall(block):
            row["time/step" if key == "step" else key] = float(value)
        rows.append(row)
    return rows


def parse_resources(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for raw in reader:
            rows.append({key: float(value) for key, value in raw.items()})
    return rows


def finite_values(rows: list[dict[str, float]], key: str) -> list[float]:
    return [row[key] for row in rows if key in row and math.isfinite(row[key])]


def rolling(values: list[float], window: int) -> list[float]:
    result = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result.append(mean(values[start : index + 1]))
    return result


FONT_REGULAR = Path(r"C:\Windows\Fonts\segoeui.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\segoeuib.ttf")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


COLORS = {
    "bg": "#ffffff",
    "fg": "#172033",
    "muted": "#657089",
    "grid": "#dce2ea",
    "blue": "#2563eb",
    "orange": "#ea580c",
    "green": "#15803d",
    "purple": "#7e22ce",
    "red": "#c2410c",
    "cyan": "#0891b2",
    "collect": "#e8f1ff",
    "qonly": "#fff2df",
}


def text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], value: str, size=24, bold=False, fill=None):
    draw.text(xy, value, font=font(size, bold), fill=fill or COLORS["fg"])


def line_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    x: list[float],
    series: list[tuple[str, list[float], str, int]],
    *,
    title_value: str,
    y_min: float | None = None,
    y_max: float | None = None,
    y_ticks: int = 4,
    x_label: str = "cycle",
    y_label: str = "",
    phase_boundary: float | None = None,
    percent: bool = False,
):
    left, top, right, bottom = box
    plot_left, plot_top = left + 76, top + 48
    plot_right, plot_bottom = right - 26, bottom - 55
    text(draw, (left, top), title_value, 26, True)
    values = [value for _, ys, _, _ in series for value in ys if math.isfinite(value)]
    low = min(values) if y_min is None else y_min
    high = max(values) if y_max is None else y_max
    if math.isclose(low, high):
        high = low + 1.0
    pad = 0.05 * (high - low)
    if y_min is None:
        low -= pad
    if y_max is None:
        high += pad
    x_low, x_high = min(x), max(x)
    if math.isclose(x_low, x_high):
        x_high = x_low + 1.0

    def px(value: float) -> float:
        return plot_left + (value - x_low) / (x_high - x_low) * (plot_right - plot_left)

    def py(value: float) -> float:
        return plot_bottom - (value - low) / (high - low) * (plot_bottom - plot_top)

    if phase_boundary is not None:
        boundary = px(phase_boundary)
        draw.rectangle((plot_left, plot_top, boundary, plot_bottom), fill=COLORS["collect"])
        draw.rectangle((boundary, plot_top, plot_right, plot_bottom), fill=COLORS["qonly"])
        text(draw, (plot_left + 8, plot_top + 6), "collect", 18, fill=COLORS["muted"])
        text(draw, (boundary + 8, plot_top + 6), "q-only", 18, fill=COLORS["muted"])

    for tick in range(y_ticks + 1):
        value = low + (high - low) * tick / y_ticks
        y_coord = py(value)
        draw.line((plot_left, y_coord, plot_right, y_coord), fill=COLORS["grid"], width=1)
        label = f"{value * 100:.0f}%" if percent else f"{value:.2g}"
        text(draw, (left + 4, y_coord - 13), label, 18, fill=COLORS["muted"])

    for tick in range(6):
        value = x_low + (x_high - x_low) * tick / 5
        x_coord = px(value)
        draw.line((x_coord, plot_top, x_coord, plot_bottom), fill=COLORS["grid"], width=1)
        text(draw, (x_coord - 14, plot_bottom + 10), f"{value:.0f}", 18, fill=COLORS["muted"])

    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=COLORS["fg"], width=2)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=COLORS["fg"], width=2)
    text(draw, (plot_right - 55, bottom - 30), x_label, 18, fill=COLORS["muted"])
    if y_label:
        text(draw, (left + 4, plot_top - 28), y_label, 18, fill=COLORS["muted"])

    legend_x = plot_left
    for name, ys, color, width in series:
        points = [(px(xv), py(yv)) for xv, yv in zip(x, ys) if math.isfinite(yv)]
        if len(points) >= 2:
            draw.line(points, fill=color, width=width, joint="curve")
        for point in points[:: max(1, len(points) // 16)]:
            radius = 2 if width < 4 else 3
            draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill=color)
        draw.line((legend_x, top + 34, legend_x + 26, top + 34), fill=color, width=width)
        text(draw, (legend_x + 34, top + 21), name, 18)
        legend_x += 34 + draw.textlength(name, font=font(18)) + 42


metrics = parse_metric_log(INPUT / "metrics.log")
resources = parse_resources(INPUT / "resources.csv")
if not metrics:
    raise SystemExit("no metric rows parsed")

steps = [int(row["cycle"]) for row in metrics]
success = [row.get("success_once", float("nan")) for row in metrics]
episodes = [row.get("num_trajectories", 0.0) for row in metrics]
success_count = sum(s * n for s, n in zip(success, episodes) if math.isfinite(s))
episode_count = sum(episodes)
first_q_row = next((row for row in metrics if row.get("qam/critic_updates", 0.0) > 0), None)
first_q_step = int(first_q_row["cycle"]) if first_q_row else None
last = metrics[-1]

resource_start = resources[0]["unix_time"]
resource_minutes = [(row["unix_time"] - resource_start) / 60.0 for row in resources]
active = [row for row in resources if row["compute_process_count"] > 0]

summary = {
    "latest_completed_step": int(last["cycle"]),
    "step_ceiling": int(last["step_ceiling"]),
    "logged_episodes": int(episode_count),
    "logged_successes": float(success_count),
    "logged_success_rate": float(success_count / episode_count),
    "last_10_success_rate": float(
        sum(s * n for s, n in zip(success[-10:], episodes[-10:]))
        / sum(episodes[-10:])
    ),
    "last_20_success_rate": float(
        sum(s * n for s, n in zip(success[-20:], episodes[-20:]))
        / sum(episodes[-20:])
    ),
    "first_q_step": first_q_step,
    "latest_global_total_inserts": int(last.get("qam/global_total_inserts", 0)),
    "latest_local_replay_mean": float(last.get("qam/local_replay_size", 0)),
    "latest_critic_updates": int(last.get("qam/critic_updates", 0)),
    "latest_fine_updates": int(last.get("qam/fine_updates", 0)),
    "latest_policy_version": int(last.get("qam/fine_policy_version", 0)),
    "latest_critic_loss": last.get("qam/critic_loss"),
    "latest_critic_grad_norm": last.get("qam/critic_grad_norm"),
    "latest_q_mean": last.get("qam/q_mean"),
    "latest_q_std_heads": last.get("qam/q_std_heads"),
    "latest_td_target_mean": last.get("qam/td_target_mean"),
    "critic_loss_min": min(finite_values(metrics, "qam/critic_loss")),
    "critic_loss_max": max(finite_values(metrics, "qam/critic_loss")),
    "critic_grad_max": max(finite_values(metrics, "qam/critic_grad_norm")),
    "resource_samples": len(resources),
    "resource_duration_seconds": int(resources[-1]["unix_time"] - resource_start),
    "gpu0_peak_mib": int(max(row["gpu0_used_mib"] for row in resources)),
    "gpu1_peak_mib": int(max(row["gpu1_used_mib"] for row in resources)),
    "gpu0_active_mean_util_pct": mean(row["gpu0_util_pct"] for row in active),
    "gpu1_active_mean_util_pct": mean(row["gpu1_util_pct"] for row in active),
    "cgroup_current_peak_gib": max(row["cgroup_current_bytes"] for row in resources) / 2**30,
    "cgroup_anon_peak_gib": max(row["cgroup_anon_bytes"] for row in resources) / 2**30,
    "cgroup_file_peak_gib": max(row["cgroup_file_bytes"] for row in resources) / 2**30,
    "qam_process_rss_peak_gib": max(row["qam_process_rss_kib"] for row in resources) / 2**20,
    "disk_delta_gib": (resources[0]["disk_available_bytes"] - resources[-1]["disk_available_bytes"]) / 2**30,
    "oom_events_max": int(max(row["cgroup_oom_events"] for row in resources)),
    "oom_kill_events_max": int(max(row["cgroup_oom_kill_events"] for row in resources)),
}

(INPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(INPUT / "steps.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
(INPUT / "resources.json").write_text(json.dumps(resources, indent=2), encoding="utf-8")


# Success chart.
image = Image.new("RGB", (1400, 760), COLORS["bg"])
draw = ImageDraw.Draw(image)
roll10 = rolling(success, 10)
roll20 = rolling(success, 20)
line_chart(
    draw,
    (40, 40, 1360, 700),
    [float(value) for value in steps],
    [
        ("per-cycle", success, COLORS["cyan"], 2),
        ("rolling-10", roll10, COLORS["blue"], 5),
        ("rolling-20", roll20, COLORS["orange"], 4),
    ],
    title_value="Train success through latest completed cycle",
    y_min=0.0,
    y_max=1.0,
    phase_boundary=float(first_q_step or steps[-1]) - 0.5,
    percent=True,
)
text(
    draw,
    (92, 710),
    f"Logged: {success_count:.0f}/{episode_count:.0f} successes ({success_count/episode_count:.1%}); "
    f"last 10 cycles {summary['last_10_success_rate']:.1%}; run failed before cycle 52 metrics were emitted.",
    22,
)
image.save(OUTPUT / "qam-success.png")


# Optimization chart: four aligned facets.
image = Image.new("RGB", (1400, 1120), COLORS["bg"])
draw = ImageDraw.Draw(image)
q_rows = [row for row in metrics if "qam/critic_loss" in row]
q_steps = [row["cycle"] for row in q_rows]
facets = [
    ("Critic loss", "qam/critic_loss", COLORS["blue"], 0.0, None),
    ("Critic grad norm (pre-clip)", "qam/critic_grad_norm", COLORS["orange"], 0.0, None),
    ("Q mean and TD target", None, COLORS["green"], None, None),
    ("10-Q head disagreement", "qam/q_std_heads", COLORS["purple"], 0.0, None),
]
boxes = [(30, 30, 690, 540), (710, 30, 1370, 540), (30, 570, 690, 1080), (710, 570, 1370, 1080)]
for box, facet in zip(boxes, facets):
    title_value, key, color, low, high = facet
    if key is None:
        series = [
            ("Q mean", [row["qam/q_mean"] for row in q_rows], COLORS["green"], 4),
            ("TD target", [row["qam/td_target_mean"] for row in q_rows], COLORS["red"], 3),
        ]
    else:
        series = [(title_value, [row[key] for row in q_rows], color, 4)]
    line_chart(
        draw,
        box,
        q_steps,
        series,
        title_value=title_value,
        y_min=low,
        y_max=high,
        phase_boundary=None,
    )
image.save(OUTPUT / "qam-optimization.png")


# Resource chart: GPU and cgroup facets.
image = Image.new("RGB", (1400, 1120), COLORS["bg"])
draw = ImageDraw.Draw(image)
line_chart(
    draw,
    (30, 30, 1370, 540),
    resource_minutes,
    [
        ("GPU0 memory GiB", [row["gpu0_used_mib"] / 1024 for row in resources], COLORS["blue"], 4),
        ("GPU1 memory GiB", [row["gpu1_used_mib"] / 1024 for row in resources], COLORS["orange"], 4),
    ],
    title_value="GPU memory",
    y_min=0.0,
    y_max=80.0,
    x_label="minutes",
    y_label="GiB",
)
line_chart(
    draw,
    (30, 570, 1370, 1080),
    resource_minutes,
    [
        ("cgroup current", [row["cgroup_current_bytes"] / 2**30 for row in resources], COLORS["purple"], 4),
        ("anonymous", [row["cgroup_anon_bytes"] / 2**30 for row in resources], COLORS["green"], 4),
        ("file cache", [row["cgroup_file_bytes"] / 2**30 for row in resources], COLORS["cyan"], 3),
    ],
    title_value="Host cgroup memory",
    y_min=0.0,
    y_max=240.0,
    x_label="minutes",
    y_label="GiB",
)
image.save(OUTPUT / "qam-resources.png")

print(json.dumps(summary, indent=2))
print(f"OUTPUT={OUTPUT}")
