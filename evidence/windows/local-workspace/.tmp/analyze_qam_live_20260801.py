from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, median

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"C:\Users\86136\Documents\rl")
SNAPSHOT = ROOT / ".tmp" / "qam_status_20260801_1158"
VISUALS = Path(
    r"E:\Codex\home\visualizations\2026\07\28\019fa85b-0b0c-72d0-a9ea-a9d88c48e8aa"
)
ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
KV_RE = re.compile(rf"([A-Za-z_][A-Za-z0-9_./-]*)=({NUMBER})")
STEP_RE = re.compile(r"Global Step:\s*(\d+)/(\d+)")
STEP_TIME_RE = re.compile(rf"Step Time:\s*({NUMBER})s")


def parse_log(path: Path, low: int, high: int) -> list[dict[str, float]]:
    text = ANSI.sub("", path.read_text(encoding="utf-8", errors="replace"))
    matches = list(STEP_RE.finditer(text))
    rows: list[dict[str, float]] = []
    for index, match in enumerate(matches):
        cycle = int(match.group(1))
        if not low <= cycle <= high:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.start() : end]
        row: dict[str, float] = {
            "cycle": float(cycle),
            "step_ceiling": float(match.group(2)),
        }
        step_time = STEP_TIME_RE.search(block)
        if step_time:
            row["step_time_seconds"] = float(step_time.group(1))
        for key, value in KV_RE.findall(block):
            row["time/step" if key == "step" else key] = float(value)
        rows.append(row)
    return rows


def parse_resources(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            try:
                rows.append({key: float(value) for key, value in raw.items()})
            except (TypeError, ValueError):
                continue
    return rows


def success_rate(rows: list[dict[str, float]]) -> float:
    episodes = sum(row.get("num_trajectories", 0.0) for row in rows)
    successes = sum(
        row.get("success_once", 0.0) * row.get("num_trajectories", 0.0)
        for row in rows
    )
    return successes / episodes if episodes else float("nan")


def values(rows: list[dict[str, float]], key: str) -> list[float]:
    return [row[key] for row in rows if key in row and math.isfinite(row[key])]


def rolling_success(rows: list[dict[str, float]], window: int) -> list[float]:
    return [success_rate(rows[max(0, i - window + 1) : i + 1]) for i in range(len(rows))]


metrics = (
    parse_log(SNAPSHOT / "v2_driver.log", 1, 25)
    + parse_log(SNAPSHOT / "v3_driver.log", 26, 100)
    + parse_log(SNAPSHOT / "v4_driver.log", 101, 380)
)
metrics.sort(key=lambda row: row["cycle"])
resources = parse_resources(SNAPSHOT / "v4_resources.csv")
if not metrics or not resources:
    raise SystemExit("missing metrics or resources")

latest = metrics[-1]
q_rows = [row for row in metrics if "qam/critic_loss" in row]
am_rows = [row for row in metrics if row.get("qam/fine_updates", 0.0) > 0]
stage_rows = {
    "collect": [row for row in metrics if row["cycle"] <= 25],
    "q_only": [row for row in metrics if 26 <= row["cycle"] <= 51],
    "am_on": [row for row in metrics if row["cycle"] >= 52],
}
active = [row for row in resources if row["compute_process_count"] > 0]
resource_minutes = [
    (row["unix_time"] - resources[0]["unix_time"]) / 60.0 for row in resources
]
step_times = values(metrics[-20:], "step_time_seconds")

summary = {
    "snapshot_cycle": int(latest["cycle"]),
    "endpoint": int(latest["step_ceiling"]),
    "cycles_parsed": len(metrics),
    "successes": sum(
        row.get("success_once", 0.0) * row.get("num_trajectories", 0.0)
        for row in metrics
    ),
    "episodes": sum(row.get("num_trajectories", 0.0) for row in metrics),
    "success_rate": success_rate(metrics),
    "last_10_success_rate": success_rate(metrics[-10:]),
    "last_20_success_rate": success_rate(metrics[-20:]),
    "last_50_success_rate": success_rate(metrics[-50:]),
    "stage_success": {key: success_rate(rows) for key, rows in stage_rows.items()},
    "latest": {
        key: latest.get(key)
        for key in (
            "qam/global_total_inserts",
            "qam/critic_updates",
            "qam/fine_updates",
            "qam/fine_policy_version",
            "qam/critic_loss",
            "qam/critic_grad_norm",
            "qam/q_mean",
            "qam/q_std_heads",
            "qam/td_target_mean",
            "qam/am_loss",
            "qam/terminal_adjoint_norm",
            "qam/fine_grad_norm",
            "qam/pending_update_credit",
        )
    },
    "q_recent_20": {
        key: mean(values(q_rows[-20:], key))
        for key in (
            "qam/critic_loss",
            "qam/critic_grad_norm",
            "qam/q_mean",
            "qam/q_std_heads",
            "qam/td_target_mean",
        )
    },
    "am_first_10": {
        key: mean(values(am_rows[:10], key))
        for key in (
            "qam/am_loss",
            "qam/terminal_adjoint_norm",
            "qam/fine_grad_norm",
        )
    },
    "am_recent_10": {
        key: mean(values(am_rows[-10:], key))
        for key in (
            "qam/am_loss",
            "qam/terminal_adjoint_norm",
            "qam/fine_grad_norm",
        )
    },
    "median_step_time_recent_20_seconds": median(step_times),
    "gpu": {
        "current_mib": [resources[-1]["gpu0_used_mib"], resources[-1]["gpu1_used_mib"]],
        "peak_mib": [
            max(row["gpu0_used_mib"] for row in resources),
            max(row["gpu1_used_mib"] for row in resources),
        ],
        "active_mean_util_pct": [
            mean(row["gpu0_util_pct"] for row in active),
            mean(row["gpu1_util_pct"] for row in active),
        ],
    },
    "host": {
        "current_gib": resources[-1]["cgroup_current_bytes"] / 2**30,
        "anon_current_gib": resources[-1]["cgroup_anon_bytes"] / 2**30,
        "file_current_gib": resources[-1]["cgroup_file_bytes"] / 2**30,
        "current_peak_gib": max(row["cgroup_current_bytes"] for row in resources) / 2**30,
        "anon_peak_gib": max(row["cgroup_anon_bytes"] for row in resources) / 2**30,
        "file_peak_gib": max(row["cgroup_file_bytes"] for row in resources) / 2**30,
        "qam_rss_peak_gib": max(row["qam_process_rss_kib"] for row in resources) / 2**20,
        "oom": int(max(row["cgroup_oom_events"] for row in resources)),
        "oom_kill": int(max(row["cgroup_oom_kill_events"] for row in resources)),
    },
    "resource_duration_minutes": resource_minutes[-1],
}


def compact(rows: list[dict[str, float]], keys: tuple[str, ...], stride: int) -> list[list[float | None]]:
    selected = rows[::stride]
    if selected and selected[-1] is not rows[-1]:
        selected.append(rows[-1])
    return [
        [round(row["cycle"], 3)]
        + [round(row[key], 6) if key in row and math.isfinite(row[key]) else None for key in keys]
        for row in selected
    ]


roll20 = rolling_success(metrics, 20)
success_compact = []
for index in range(0, len(metrics), 4):
    row = metrics[index]
    success_compact.append(
        [int(row["cycle"]), round(row.get("success_once", 0.0), 3), round(roll20[index], 4)]
    )
if success_compact[-1][0] != int(metrics[-1]["cycle"]):
    success_compact.append(
        [int(metrics[-1]["cycle"]), round(metrics[-1].get("success_once", 0.0), 3), round(roll20[-1], 4)]
    )

resource_compact = []
for index in range(0, len(resources), 30):
    row = resources[index]
    resource_compact.append(
        [
            round(resource_minutes[index], 2),
            round(row["gpu0_used_mib"] / 1024, 2),
            round(row["gpu1_used_mib"] / 1024, 2),
            round(row["cgroup_anon_bytes"] / 2**30, 2),
            round(row["cgroup_file_bytes"] / 2**30, 2),
        ]
    )
if resource_compact[-1][0] != round(resource_minutes[-1], 2):
    row = resources[-1]
    resource_compact.append(
        [
            round(resource_minutes[-1], 2),
            round(row["gpu0_used_mib"] / 1024, 2),
            round(row["gpu1_used_mib"] / 1024, 2),
            round(row["cgroup_anon_bytes"] / 2**30, 2),
            round(row["cgroup_file_bytes"] / 2**30, 2),
        ]
    )

compact_data = {
    "success": success_compact,
    "optimization": compact(
        q_rows,
        (
            "qam/critic_loss",
            "qam/q_mean",
            "qam/td_target_mean",
            "qam/q_std_heads",
        ),
        3,
    ),
    "am": compact(
        am_rows,
        ("qam/am_loss", "qam/terminal_adjoint_norm", "qam/fine_grad_norm"),
        3,
    ),
    "resources": resource_compact,
}

SNAPSHOT.joinpath("analysis.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
SNAPSHOT.joinpath("compact.json").write_text(
    json.dumps(compact_data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
)


def load_font(size: int, bold: bool = False):
    path = Path(r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc")
    return ImageFont.truetype(str(path), size=size) if path.exists() else ImageFont.load_default()


COLORS = {
    "bg": "#ffffff",
    "fg": "#172033",
    "muted": "#657089",
    "grid": "#dce2ea",
    "blue": "#2563eb",
    "orange": "#ea580c",
    "green": "#15803d",
    "purple": "#7e22ce",
    "cyan": "#0891b2",
    "red": "#c2410c",
    "collect": "#e8f1ff",
    "qonly": "#fff2df",
}


def draw_text(draw, xy, value, size=24, bold=False, fill=None):
    draw.text(xy, value, font=load_font(size, bold), fill=fill or COLORS["fg"])


def line_chart(draw, box, xs, series, title, low=None, high=None, phase=False, percent=False):
    left, top, right, bottom = box
    pl, pt, pr, pb = left + 90, top + 55, right - 25, bottom - 55
    draw_text(draw, (left, top), title, 27, True)
    ys_all = [v for _, ys, _ in series for v in ys if math.isfinite(v)]
    lo = min(ys_all) if low is None else low
    hi = max(ys_all) if high is None else high
    if math.isclose(lo, hi):
        hi = lo + 1
    pad = (hi - lo) * 0.05
    if low is None:
        lo -= pad
    if high is None:
        hi += pad
    x0, x1 = min(xs), max(xs)
    px = lambda x: pl + (x - x0) / (x1 - x0) * (pr - pl)
    py = lambda y: pb - (y - lo) / (hi - lo) * (pb - pt)
    if phase:
        draw.rectangle((pl, pt, px(25.5), pb), fill=COLORS["collect"])
        draw.rectangle((px(25.5), pt, px(51.5), pb), fill=COLORS["qonly"])
    for tick in range(5):
        value = lo + (hi - lo) * tick / 4
        yy = py(value)
        draw.line((pl, yy, pr, yy), fill=COLORS["grid"], width=1)
        label = f"{value:.0%}" if percent else f"{value:.3g}"
        draw_text(draw, (left, yy - 12), label, 18, fill=COLORS["muted"])
    for name, ys, color in series:
        points = [(px(x), py(y)) for x, y in zip(xs, ys) if math.isfinite(y)]
        if len(points) > 1:
            draw.line(points, fill=color, width=4, joint="curve")
    legend_x = pl
    for name, _, color in series:
        draw.line((legend_x, top + 38, legend_x + 30, top + 38), fill=color, width=4)
        draw_text(draw, (legend_x + 38, top + 24), name, 18)
        legend_x += 38 + int(draw.textlength(name, font=load_font(18))) + 38
    draw.line((pl, pb, pr, pb), fill=COLORS["fg"], width=2)
    draw.line((pl, pt, pl, pb), fill=COLORS["fg"], width=2)
    for tick in range(5):
        x = x0 + (x1 - x0) * tick / 4
        draw_text(draw, (px(x) - 18, pb + 10), f"{x:.0f}", 18, fill=COLORS["muted"])


VISUALS.mkdir(parents=True, exist_ok=True)
img = Image.new("RGB", (1200, 1580), COLORS["bg"])
draw = ImageDraw.Draw(img)
draw_text(draw, (45, 25), f"QAM 正式训练快照 · cycle {int(latest['cycle'])}/380", 34, True)
draw_text(
    draw,
    (45, 72),
    f"成功 {summary['successes']:.0f}/{summary['episodes']:.0f} ({summary['success_rate']:.1%}) · "
    f"critic {int(latest['qam/critic_updates'])} · fine {int(latest['qam/fine_updates'])}",
    23,
    fill=COLORS["muted"],
)

success_x = [row["cycle"] for row in metrics]
success_raw = [row.get("success_once", 0.0) for row in metrics]
success_roll = rolling_success(metrics, 20)
line_chart(
    draw,
    (35, 125, 1165, 520),
    success_x,
    [("每 cycle", success_raw, COLORS["cyan"]), ("滚动20", success_roll, COLORS["blue"])],
    "训练成功率",
    low=0.0,
    high=1.0,
    phase=True,
    percent=True,
)

q_x = [row["cycle"] for row in q_rows]
line_chart(
    draw,
    (35, 555, 1165, 935),
    q_x,
    [
        ("Q mean", [row["qam/q_mean"] for row in q_rows], COLORS["green"]),
        ("TD target", [row["qam/td_target_mean"] for row in q_rows], COLORS["red"]),
    ],
    "Q 与 TD target",
)

minutes = resource_minutes
line_chart(
    draw,
    (35, 970, 1165, 1350),
    minutes,
    [
        ("GPU0 GiB", [row["gpu0_used_mib"] / 1024 for row in resources], COLORS["blue"]),
        ("GPU1 GiB", [row["gpu1_used_mib"] / 1024 for row in resources], COLORS["orange"]),
        ("host anon GiB", [row["cgroup_anon_bytes"] / 2**30 for row in resources], COLORS["purple"]),
    ],
    "v4 资源",
    low=0.0,
    high=80.0,
)
draw_text(
    draw,
    (45, 1390),
    f"最新：critic loss {latest['qam/critic_loss']:.4f} · AM loss {latest['qam/am_loss']:.3f} · "
    f"Q std {latest['qam/q_std_heads']:.3f}",
    22,
)
draw_text(
    draw,
    (45, 1430),
    f"GPU peak {summary['gpu']['peak_mib'][0]/1024:.1f}/{summary['gpu']['peak_mib'][1]/1024:.1f} GiB · "
    f"host anon peak {summary['host']['anon_peak_gib']:.1f} GiB · OOM/OOM-kill 0/0",
    22,
)
draw_text(
    draw,
    (45, 1480),
    "浅蓝=collect，浅橙=q_only，之后=am_on；成功率每点仅2个episode，短窗波动较大。",
    19,
    fill=COLORS["muted"],
)
png = VISUALS / f"qam-live-cycle-{int(latest['cycle'])}.png"
img.save(png)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print("COMPACT=" + json.dumps(compact_data, ensure_ascii=False, separators=(",", ":")))
print(f"PNG={png}")
