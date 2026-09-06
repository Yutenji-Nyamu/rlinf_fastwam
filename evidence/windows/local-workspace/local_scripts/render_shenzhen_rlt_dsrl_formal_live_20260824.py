#!/usr/bin/env python3
"""Parse and render the 2026-08-24 Shenzhen RLT/DSRL formal snapshot.

The script only consumes the downloaded lightweight evidence. It does not
interpolate evaluation points, invent warm-up optimizer metrics, or contact the
server. Static PNGs are rendered with Pillow for reliable mobile viewing.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"C:\Users\86136\Documents\rl")
EVIDENCE = ROOT / "docs" / "rlinf-shenzhen-rlt-dsrl-port" / "evidence" / "formal-live-analysis-20260824"

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

INK = "#142033"
MUTED = "#64748B"
GRID = "#D8E0EA"
LIGHT = "#F4F7FB"
WHITE = "#FFFFFF"
BLUE = "#2563EB"
BLUE_LIGHT = "#93C5FD"
ORANGE = "#EA580C"
ORANGE_LIGHT = "#FDBA74"
GREEN = "#16803A"
GREEN_LIGHT = "#86EFAC"
PURPLE = "#7C3AED"
PURPLE_LIGHT = "#C4B5FD"
RED = "#DC2626"
RED_LIGHT = "#FCA5A5"
TEAL = "#0F766E"
GRAY = "#94A3B8"
STALL_FILL = "#FEF2F2"

CST = timezone(timedelta(hours=8))
STALL_UTC = datetime.fromisoformat("2026-08-23T17:39:19+00:00")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path(r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


F_TITLE = font(43, True)
F_SUB = font(23)
F_PANEL = font(29, True)
F_AXIS = font(19)
F_LEGEND = font(19, True)
F_CARD = font(24, True)
F_CARD_SMALL = font(17)
F_NOTE = font(18)


def clean_text(path: Path) -> str:
    return ANSI_RE.sub("", path.read_text(encoding="utf-8", errors="replace")).replace("\r", "\n")


def optional_number(segment: str, key: str) -> float | None:
    match = re.search(rf"(?<![\w/]){re.escape(key)}=({NUMBER})", segment)
    return float(match.group(1)) if match else None


def parse_elapsed(value: str) -> float:
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(value)


def parse_stage1(path: Path) -> list[dict[str, float | int]]:
    rows: dict[int, dict[str, float | int]] = {}
    for line in clean_text(path).splitlines():
        if "train/loss=" not in line or "/2000" not in line:
            continue
        step_match = re.search(r"\|\s*(\d+)/2000\s+\[", line)
        if not step_match:
            continue
        step = int(step_match.group(1))
        elapsed_match = re.search(r"\[([0-9:]+)<", line)
        row: dict[str, float | int] = {"step": step}
        if elapsed_match:
            row["elapsed_s"] = parse_elapsed(elapsed_match.group(1))
        for key, out_key in [
            ("time/step", "step_time_s"),
            ("time/training", "training_time_s"),
            ("train/grad_norm", "grad_norm"),
            ("train/learning_rate", "learning_rate"),
            ("train/loss", "loss"),
            ("train/rlt_loss", "rlt_loss"),
            ("train/vla_loss", "vla_loss"),
        ]:
            value = optional_number(line, key)
            if value is not None:
                row[out_key] = value
        if {"loss", "grad_norm", "learning_rate", "step_time_s"}.issubset(row):
            rows[step] = row
    ordered = [rows[step] for step in sorted(rows)]
    steps = [int(row["step"]) for row in ordered]
    if steps != list(range(1, 2001)):
        raise ValueError(f"RLT Stage1 steps are not 1..2000: {steps[:3]} ... {steps[-3:]}")
    return ordered


def parse_metric_tables(path: Path, total_steps: int, algorithm: str) -> list[dict[str, float | int | None]]:
    text = clean_text(path)
    matches = list(re.finditer(rf"Global Step:\s*(\d+)/{total_steps}", text))
    rows: list[dict[str, float | int | None]] = []
    for index, match in enumerate(matches):
        step = int(match.group(1))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segment = text[match.start() : end]
        success = [float(value) for value in re.findall(rf"(?<![\w/])success_once=({NUMBER})", segment)]
        elapsed_match = re.search(r"Elapsed:\s*([0-9:]+)", segment)
        step_time_match = re.search(rf"Step Time:\s*({NUMBER})s", segment)
        if not success or not elapsed_match or not step_time_match:
            continue
        row: dict[str, float | int | None] = {
            "step": step,
            "elapsed_s": parse_elapsed(elapsed_match.group(1)),
            "step_time_s": float(step_time_match.group(1)),
            "train_success": success[0],
            "eval_success": success[1] if len(success) > 1 else None,
            "reward": optional_number(segment, "reward"),
            "generate_rollouts_s": optional_number(segment, "generate_rollouts"),
            "actor_training_s": optional_number(segment, "actor_training"),
        }
        if algorithm == "rlt":
            row.update(
                {
                    "local_cache_size": optional_number(segment, "cache_size"),
                    "local_transition_count": optional_number(segment, "replay/transition_count"),
                    "global_min_replay_size": optional_number(segment, "rlt/global_min_replay_size"),
                    "update_step": optional_number(segment, "rlt/update_step"),
                    "ready_for_online": optional_number(segment, "rlt/ready_for_online"),
                    "actor_loss": optional_number(segment, "sac/actor_loss"),
                    "critic_loss": optional_number(segment, "sac/critic_loss"),
                }
            )
        else:
            row.update(
                {
                    "local_resident_transitions": optional_number(segment, "resident_transitions"),
                    "global_new_transitions": optional_number(segment, "sac/global_new_transitions"),
                    "global_resident_transitions": optional_number(segment, "sac/global_resident_transitions"),
                    "planned_optimizer_updates": optional_number(segment, "sac/planned_optimizer_updates"),
                    "actor_loss": optional_number(segment, "sac/actor_loss"),
                    "critic_loss": optional_number(segment, "sac/critic_loss"),
                    "alpha": optional_number(segment, "sac/alpha"),
                    "alpha_loss": optional_number(segment, "sac/alpha_loss"),
                    "actor_entropy": optional_number(segment, "actor/entropy"),
                    "actor_grad_norm": optional_number(segment, "actor/grad_norm"),
                    "critic_grad_norm": optional_number(segment, "critic/grad_norm"),
                    "actor_q_pi": optional_number(segment, "actor/q_pi"),
                    "critic_q_data": optional_number(segment, "critic/q_data"),
                }
            )
        rows.append(row)
    deduped: dict[int, dict[str, float | int | None]] = {int(row["step"]): row for row in rows}
    ordered = [deduped[step] for step in sorted(deduped)]
    steps = [int(row["step"]) for row in ordered]
    if not steps or steps != list(range(1, max(steps) + 1)):
        raise ValueError(f"{algorithm} steps are not consecutive: {steps[:3]} ... {steps[-3:]}")
    return ordered


def read_resource(path: Path, gpu_ids: tuple[int, int]) -> list[dict[str, float | datetime]]:
    def optional_float(value: str | None) -> float:
        text = (value or "").strip()
        return float(text) if text else math.nan

    rows: list[dict[str, float | datetime]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, float | datetime] = {
                "timestamp": datetime.fromisoformat(raw["timestamp"]),
                "host_available_gib": float(raw["host_mem_available_kib"]) / 1024.0 / 1024.0,
                # The observer writes one terminal row after a driver exits.  That
                # row intentionally keeps the host-memory sample while leaving
                # dead-cgroup counters and GPU columns empty.
                "cgroup_oom": optional_float(raw.get("cgroup_oom")),
                "cgroup_oom_kill": optional_float(raw.get("cgroup_oom_kill")),
            }
            for gpu_id in gpu_ids:
                used = raw.get(f"gpu{gpu_id}_used_mib", "").strip()
                util = raw.get(f"gpu{gpu_id}_util_pct", "").strip()
                row[f"gpu{gpu_id}_used_gib"] = float(used) / 1024.0 if used else math.nan
                row[f"gpu{gpu_id}_util_pct"] = float(util) if util else math.nan
            rows.append(row)
    return rows


def rolling(values: list[float | None], window: int) -> list[float | None]:
    out: list[float | None] = []
    for index in range(len(values)):
        chunk = values[max(0, index - window + 1) : index + 1]
        finite = [float(value) for value in chunk if value is not None and math.isfinite(float(value))]
        out.append(statistics.mean(finite) if len(finite) == window else None)
    return out


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty percentile input")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def text_width(draw: ImageDraw.ImageDraw, value: str, fnt: ImageFont.FreeTypeFont) -> int:
    box = draw.textbbox((0, 0), value, font=fnt)
    return box[2] - box[0]


def add_header(image: Image.Image, title: str, subtitle: str) -> ImageDraw.ImageDraw:
    draw = ImageDraw.Draw(image)
    draw.text((58, 34), title, fill=INK, font=F_TITLE)
    draw.text((60, 94), subtitle, fill=MUTED, font=F_SUB)
    return draw


def draw_cards(draw: ImageDraw.ImageDraw, y: int, cards: list[tuple[str, str, str]], width: int = 1600) -> int:
    gap = 18
    left = 55
    card_width = (width - 110 - gap * (len(cards) - 1)) // len(cards)
    for index, (title, value, note) in enumerate(cards):
        x0 = left + index * (card_width + gap)
        x1 = x0 + card_width
        draw.rounded_rectangle((x0, y, x1, y + 118), radius=18, fill=WHITE, outline=GRID, width=2)
        draw.text((x0 + 18, y + 14), title, font=F_CARD_SMALL, fill=MUTED)
        draw.text((x0 + 18, y + 41), value, font=F_CARD, fill=INK)
        draw.text((x0 + 18, y + 84), note, font=F_CARD_SMALL, fill=MUTED)
    return y + 140


Series = tuple[str, list[float | None], str, int, bool]


def finite_values(series: Iterable[Series]) -> list[float]:
    return [float(value) for _, values, _, _, _ in series for value in values if value is not None and math.isfinite(float(value))]


def draw_line_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    xs: list[float],
    series: list[Series],
    title: str,
    *,
    y_min: float | None = None,
    y_max: float | None = None,
    y_formatter: Callable[[float], str] | None = None,
    x_formatter: Callable[[float], str] | None = None,
    x_ticks: list[float] | None = None,
    verticals: list[tuple[float, str, str]] | None = None,
    shades: list[tuple[float, float, str, str]] | None = None,
    zero_line: bool = False,
) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=20, fill=WHITE, outline=GRID, width=2)
    draw.text((left + 24, top + 18), title, fill=INK, font=F_PANEL)
    plot_left, plot_top = left + 112, top + 82
    plot_right, plot_bottom = right - 35, bottom - 70
    x_min, x_max = min(xs), max(xs)
    if math.isclose(x_min, x_max):
        x_max = x_min + 1
    values = finite_values(series)
    if not values:
        raise ValueError(f"no finite values for {title}")
    raw_min, raw_max = min(values), max(values)
    span = max(1e-9, raw_max - raw_min)
    if y_min is None:
        y_min = raw_min - span * 0.12
    if y_max is None:
        y_max = raw_max + span * 0.15
    if math.isclose(y_min, y_max):
        y_min -= 1
        y_max += 1

    def px(value: float) -> float:
        return plot_left + (value - x_min) / (x_max - x_min) * (plot_right - plot_left)

    def py(value: float) -> float:
        return plot_bottom - (value - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

    if shades:
        for start, end, fill, label in shades:
            start = max(start, x_min)
            end = min(end, x_max)
            if end > start:
                draw.rectangle((px(start), plot_top, px(end), plot_bottom), fill=fill)
                draw.text((px(start) + 8, plot_top + 8), label, font=F_NOTE, fill=RED)

    for index in range(6):
        value = y_min + (y_max - y_min) * index / 5
        y = py(value)
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=2)
        label = y_formatter(value) if y_formatter else (f"{value:.2f}" if abs(y_max - y_min) < 5 else f"{value:.0f}")
        draw.text((plot_left - 14 - text_width(draw, label, F_AXIS), y - 11), label, fill=MUTED, font=F_AXIS)
    if zero_line and y_min < 0 < y_max:
        draw.line((plot_left, py(0), plot_right, py(0)), fill=GRAY, width=3)

    if x_ticks is None:
        x_ticks = [x_min + (x_max - x_min) * index / 5 for index in range(6)]
    for value in x_ticks:
        x = px(value)
        draw.line((x, plot_top, x, plot_bottom), fill=GRID, width=1)
        label = x_formatter(value) if x_formatter else f"{value:.0f}"
        draw.text((x - text_width(draw, label, F_AXIS) // 2, plot_bottom + 14), label, fill=MUTED, font=F_AXIS)

    if verticals:
        for value, label, color in verticals:
            if x_min <= value <= x_max:
                x = px(value)
                draw.line((x, plot_top, x, plot_bottom), fill=color, width=3)
                draw.text((x + 7, plot_top + 33), label, font=F_NOTE, fill=color)

    legend_x = left + 24
    legend_y = bottom - 39
    for label, ys, color, width, markers_only in series:
        if markers_only:
            draw.ellipse((legend_x, legend_y + 2, legend_x + 18, legend_y + 20), fill=color, outline=WHITE, width=2)
        else:
            draw.line((legend_x, legend_y + 11, legend_x + 28, legend_y + 11), fill=color, width=width)
        draw.text((legend_x + 38, legend_y - 2), label, fill=INK, font=F_LEGEND)
        legend_x += 55 + text_width(draw, label, F_LEGEND)

        points: list[tuple[float, float]] = []
        for x_value, y_value in zip(xs, ys):
            if y_value is None or not math.isfinite(float(y_value)):
                if len(points) > 1 and not markers_only:
                    draw.line(points, fill=color, width=width, joint="curve")
                points = []
                continue
            point = (px(float(x_value)), py(float(y_value)))
            if markers_only:
                draw.ellipse((point[0] - 6, point[1] - 6, point[0] + 6, point[1] + 6), fill=color, outline=WHITE, width=2)
            else:
                points.append(point)
        if len(points) > 1 and not markers_only:
            draw.line(points, fill=color, width=width, joint="curve")

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=GRAY, width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=GRAY, width=2)


def render_stage1(rows: list[dict[str, float | int]], resource: list[dict[str, float | datetime]]) -> None:
    xs = [float(row["step"]) for row in rows]
    loss = [float(row["loss"]) for row in rows]
    grad = [float(row["grad_norm"]) for row in rows]
    lr = [float(row["learning_rate"]) * 1e6 for row in rows]
    times = [float(row["step_time_s"]) for row in rows]
    mem4 = [float(row["gpu4_used_gib"]) for row in resource]
    mem5 = [float(row["gpu5_used_gib"]) for row in resource]
    rx = list(range(len(resource)))
    image = Image.new("RGB", (1600, 1910), LIGHT)
    draw = add_header(image, "RLT Stage 1：current-AR 离线训练已完整完成", "2×H100，完整 1–2000 step；浅线是原值，深线是滚动均值，不隐藏波动。")
    y = draw_cards(
        draw,
        137,
        [
            ("完成", "2000 / 2000", "wrapper exit 0"),
            ("耗时", "14.0 min", f"训练环 {rows[-1]['elapsed_s']/60:.1f}m；中位 step {statistics.median(times):.3f}s"),
            ("loss", f"{loss[-1]:.3f}", f"首100均值 {statistics.mean(loss[:100]):.3f}"),
            ("GPU峰值", f"{max(max(mem4), max(mem5)):.1f} GiB", "physical GPU 4–5"),
        ],
    )
    draw_line_panel(draw, (55, y, 1545, y + 430), xs, [("raw loss", loss, BLUE_LIGHT, 2, False), ("rolling-50", rolling(loss, 50), BLUE, 5, False)], "重建损失：早期快速下降，后段稳定波动", y_min=0)
    y += 455
    draw_line_panel(draw, (55, y, 1545, y + 430), xs, [("grad norm", grad, ORANGE_LIGHT, 2, False), ("rolling-50", rolling(grad, 50), ORANGE, 5, False), ("LR × 1e6", lr, PURPLE, 4, False)], "优化量：梯度有限；学习率按计划退火", y_min=0)
    y += 455
    draw_line_panel(draw, (55, y, 1545, y + 430), xs, [("step time", times, GREEN_LIGHT, 2, False), ("rolling-50", rolling(times, 50), GREEN, 5, False)], "单步 wall time：主体约 0.35 秒；末步 checkpoint 不伪装成普通训练 step", y_min=0)
    y += 455
    draw_line_panel(draw, (55, y, 1545, y + 280), [float(v) for v in rx], [("GPU 4", mem4, BLUE, 4, False), ("GPU 5", mem5, ORANGE, 4, False)], "离线阶段显存（observer 原始 10 秒采样）", y_min=0, x_formatter=lambda value: f"{value*10/60:.0f}m")
    image.save(EVIDENCE / "01_rlt_stage1_optimization.png", optimize=True)


def render_rlt_stage2(rows: list[dict[str, float | int | None]]) -> None:
    xs = [float(row["step"]) for row in rows]
    success = [float(row["train_success"]) for row in rows]
    replay = [float(row["global_min_replay_size"]) if row["global_min_replay_size"] is not None else None for row in rows]
    update = [float(row["update_step"]) if row["update_step"] is not None else None for row in rows]
    step_time = [float(row["step_time_s"]) for row in rows]
    rollout = [float(row["generate_rollouts_s"]) if row["generate_rollouts_s"] is not None else None for row in rows]
    actor = [float(row["actor_training_s"]) if row["actor_training_s"] is not None else None for row in rows]
    image = Image.new("RGB", (1600, 1630), LIGHT)
    draw = add_header(image, "RLT Stage 2：24 个完整 warm-up cycle；Step 25 checkpoint 卡住", "进程存活不等于正常推进。首次 fixed-20 eval 已跑完，但结果表未在 checkpoint 返回前提交。")
    y = draw_cards(
        draw,
        137,
        [
            ("完整进度", "24 / 250", "Step 25 不计入"),
            ("当前阶段", "warm-up", "update_step = 0"),
            ("min-rank replay", f"{replay[-1]:.0f} / 10000", "online 门槛按最小 rank"),
            ("异常", "checkpoint hang", "约 01:39 CST 起"),
        ],
    )
    success_xs = xs + [25.0]
    draw_line_panel(draw, (55, y, 1545, y + 400), success_xs, [("train success", success + [None], BLUE_LIGHT, 2, False), ("5-cycle mean", rolling(success, 5) + [None], BLUE, 5, False), ("fixed-20 eval", [None] * len(success) + [0.0], ORANGE, 4, True)], "训练环境成功率：warm-up rollout 不是 actor 学习曲线；首个 fixed-20 eval = 0/20", y_min=0, y_max=1, y_formatter=lambda value: f"{value*100:.0f}%")
    y += 425
    draw_line_panel(draw, (55, y, 1545, y + 400), xs, [("global min replay", replay, GREEN, 5, False), ("update_step", update, PURPLE, 4, False)], "Replay 与更新状态：每个 rank 都要达到 10,000；当前预计约 cycle 135 才 online", y_min=0, y_max=10000)
    y += 425
    draw_line_panel(draw, (55, y, 1545, y + 400), xs, [("step wall", step_time, INK, 4, False), ("rollout", rollout, ORANGE, 4, False), ("actor train", actor, PURPLE, 3, False)], "每 cycle 耗时：几乎全在 simulator rollout；warm-up 无优化开销", y_min=0)
    image.save(EVIDENCE / "02_rlt_stage2_warmup_and_checkpoint_stall.png", optimize=True)


def render_dsrl_success(rows: list[dict[str, float | int | None]]) -> None:
    xs = [float(row["step"]) for row in rows]
    success = [float(row["train_success"]) for row in rows]
    eval_success = [float(row["eval_success"]) if row["eval_success"] is not None else None for row in rows]
    resident = [float(row["global_resident_transitions"]) if row["global_resident_transitions"] is not None else None for row in rows]
    planned = [float(row["planned_optimizer_updates"]) if row["planned_optimizer_updates"] is not None else None for row in rows]
    step_time = [float(row["step_time_s"]) for row in rows]
    rollout = [float(row["generate_rollouts_s"]) if row["generate_rollouts_s"] is not None else None for row in rows]
    actor_time = [float(row["actor_training_s"]) if row["actor_training_s"] is not None else None for row in rows]
    first_update = next(int(row["step"]) for row in rows if row["actor_loss"] is not None)
    eval_points = [(int(row["step"]), float(row["eval_success"])) for row in rows if row["eval_success"] is not None]
    image = Image.new("RGB", (1600, 1640), LIGHT)
    draw = add_header(image, "DSRL：训练持续正常推进，已越过 warm-up 并稳定做 SAC 更新", "2×H100 / 4 env / UTD20；成功率、真实 fixed-12 eval、replay 与耗时均从 Step 1 画到快照末点。")
    y = draw_cards(
        draw,
        137,
        [
            ("快照完整进度", f"{int(xs[-1])} / 200", "下载时最后完整表"),
            ("online 起点", f"Step {first_update}", "resident 跨 500"),
            ("最新 train", f"{success[-1]*100:.0f}%", f"最近5步 {statistics.mean(success[-5:])*100:.1f}%"),
            ("最新 fixed-12", f"{eval_points[-1][1]*100:.1f}%", f"Step {eval_points[-1][0]}"),
        ],
    )
    draw_line_panel(draw, (55, y, 1545, y + 410), xs, [("train raw", success, BLUE_LIGHT, 2, False), ("rolling-5", rolling(success, 5), BLUE, 5, False), ("rolling-20", rolling(success, 20), TEAL, 4, False), ("fixed-12 eval", eval_success, ORANGE, 4, True)], "成功率：eval 只画真实离散点，不连接、不插值", y_min=0, y_max=1, y_formatter=lambda value: f"{value*100:.0f}%", verticals=[(first_update, "learned phase", PURPLE)])
    y += 435
    draw_line_panel(draw, (55, y, 1545, y + 410), xs, [("global resident", resident, GREEN, 5, False)], "Replay resident transitions：固定容量 ring 持续增长", y_min=0, verticals=[(first_update, "warm-up 结束", PURPLE)])
    y += 435
    draw_line_panel(draw, (55, y, 1545, y + 410), xs, [("step wall", step_time, INK, 4, False), ("rollout", rollout, ORANGE, 4, False), ("SAC update", actor_time, PURPLE, 4, False)], "每步 wall time：越过 warm-up 后主要花在 UTD20 SAC 更新", y_min=0, verticals=[(first_update, "开始优化", PURPLE)])
    image.save(EVIDENCE / "03_dsrl_success_replay_and_timing.png", optimize=True)


def render_dsrl_optimization(rows: list[dict[str, float | int | None]]) -> None:
    online = [row for row in rows if row["actor_loss"] is not None]
    xs = [float(row["step"]) for row in online]
    actor_loss = [float(row["actor_loss"]) for row in online]
    critic_loss = [float(row["critic_loss"]) for row in online]
    actor_grad = [float(row["actor_grad_norm"]) for row in online]
    critic_grad = [float(row["critic_grad_norm"]) for row in online]
    alpha = [float(row["alpha"]) for row in online]
    entropy = [float(row["actor_entropy"]) for row in online]
    q_pi = [float(row["actor_q_pi"]) for row in online]
    q_data = [float(row["critic_q_data"]) for row in online]
    image = Image.new("RGB", (1600, 2300), LIGHT)
    draw = add_header(image, "DSRL 在线优化指标：全部有限，走势平滑", "只显示 learned phase；warm-up 缺失值保持为空，不补 0。不同量纲拆开画，避免小量被大梯度压扁。")
    y = draw_cards(
        draw,
        137,
        [
            ("online steps", f"{int(xs[0])}–{int(xs[-1])}", f"共 {len(xs)} 个完整点"),
            ("actor loss", f"{actor_loss[-1]:.3f}", f"最近10均值 {statistics.mean(actor_loss[-10:]):.3f}"),
            ("critic loss", f"{critic_loss[-1]:.3f}", f"最近10均值 {statistics.mean(critic_loss[-10:]):.3f}"),
            ("alpha", f"{alpha[-1]:.4f}", "持续自适应"),
        ],
    )
    draw_line_panel(draw, (55, y, 1545, y + 375), xs, [("actor loss", actor_loss, BLUE, 4, False), ("critic loss", critic_loss, ORANGE, 4, False)], "SAC losses")
    y += 400
    draw_line_panel(draw, (55, y, 1545, y + 375), xs, [("actor grad norm", actor_grad, BLUE_LIGHT, 2, False), ("rolling-10", rolling(actor_grad, 10), BLUE, 5, False)], "Actor gradient norm", y_min=0)
    y += 400
    draw_line_panel(draw, (55, y, 1545, y + 375), xs, [("critic grad norm", critic_grad, ORANGE_LIGHT, 2, False), ("rolling-10", rolling(critic_grad, 10), ORANGE, 5, False)], "Critic gradient norm", y_min=0)
    y += 400
    draw_line_panel(draw, (55, y, 1545, y + 375), xs, [("actor entropy", entropy, PURPLE, 4, False), ("Qπ", q_pi, BLUE, 4, False), ("Qdata", q_data, ORANGE, 4, False)], "策略熵与 Q 值：同为负值但保持有限、连续", zero_line=True)
    image.save(EVIDENCE / "04_dsrl_optimization_metrics.png", optimize=True)


def render_resources(rlt: list[dict[str, float | datetime]], dsrl: list[dict[str, float | datetime]]) -> None:
    all_times = [row["timestamp"] for row in rlt + dsrl]
    start = min(all_times)
    end = max(all_times)
    xs_rlt = [(row["timestamp"] - start).total_seconds() / 3600.0 for row in rlt]
    xs_dsrl = [(row["timestamp"] - start).total_seconds() / 3600.0 for row in dsrl]
    x_end = max(xs_rlt + xs_dsrl)
    stall_x = (STALL_UTC - start).total_seconds() / 3600.0
    tick_values = [x_end * index / 5 for index in range(6)]

    def time_label(value: float) -> str:
        stamp = (start + timedelta(hours=value)).astimezone(CST)
        return stamp.strftime("%H:%M")

    rlt4 = [float(row["gpu4_used_gib"]) for row in rlt]
    rlt5 = [float(row["gpu5_used_gib"]) for row in rlt]
    dsrl6 = [float(row["gpu6_used_gib"]) for row in dsrl]
    dsrl7 = [float(row["gpu7_used_gib"]) for row in dsrl]
    rlt4u = [float(row["gpu4_util_pct"]) for row in rlt]
    rlt5u = [float(row["gpu5_util_pct"]) for row in rlt]
    dsrl6u = [float(row["gpu6_util_pct"]) for row in dsrl]
    dsrl7u = [float(row["gpu7_util_pct"]) for row in dsrl]
    host_rlt = [float(row["host_available_gib"]) for row in rlt]
    host_dsrl = [float(row["host_available_gib"]) for row in dsrl]
    image = Image.new("RGB", (1600, 1900), LIGHT)
    draw = add_header(image, "双 formal 并发资源全历史：主机健康；RLT checkpoint hang 清晰可见", "横轴为北京时间；每个点均来自 observer 原始 10 秒采样，未插值。RLT 卡住后显存常驻、利用率归零。")
    y = draw_cards(
        draw,
        137,
        [
            ("RLT 当前显存", f"{rlt4[-1]:.1f}/{rlt5[-1]:.1f} GiB", "GPU 4 / 5"),
            ("DSRL 当前显存", f"{dsrl6[-1]:.1f}/{dsrl7[-1]:.1f} GiB", "GPU 6 / 7"),
            ("host available", f"{min(host_rlt + host_dsrl):.0f} GiB min", f"最新 {host_dsrl[-1]:.0f} GiB"),
            ("OOM counters", "0 / 0", "cgroup oom / kill"),
        ],
    )
    shade = [(stall_x, x_end, STALL_FILL, "RLT checkpoint hang")]
    draw_line_panel(draw, (55, y, 1545, y + 375), xs_rlt, [("GPU 4", rlt4, BLUE, 4, False), ("GPU 5", rlt5, ORANGE, 4, False)], "RLT Stage 2 GPU memory", y_min=0, x_ticks=tick_values, x_formatter=time_label, shades=shade, verticals=[(stall_x, "01:39 checkpoint", RED)])
    y += 400
    draw_line_panel(draw, (55, y, 1545, y + 375), xs_rlt, [("GPU 4 util", rlt4u, BLUE, 3, False), ("GPU 5 util", rlt5u, ORANGE, 3, False)], "RLT Stage 2 GPU utilization", y_min=0, y_max=100, x_ticks=tick_values, x_formatter=time_label, shades=shade, verticals=[(stall_x, "save start", RED)])
    y += 400
    draw_line_panel(draw, (55, y, 1545, y + 375), xs_dsrl, [("GPU 6 memory", dsrl6, BLUE, 4, False), ("GPU 7 memory", dsrl7, ORANGE, 4, False)], "DSRL GPU memory：online phase 后稳定在约 34 GiB/card", y_min=0, x_ticks=tick_values, x_formatter=time_label)
    y += 400
    draw_line_panel(draw, (55, y, 1545, y + 375), xs_dsrl, [("GPU 6 util", dsrl6u, BLUE, 2, False), ("GPU 7 util", dsrl7u, ORANGE, 2, False)], "DSRL GPU utilization：rollout / update 形成正常锯齿", y_min=0, y_max=100, x_ticks=tick_values, x_formatter=time_label)
    y += 400
    draw_line_panel(draw, (55, y, 1545, y + 375), xs_dsrl, [("host available", host_dsrl, GREEN, 4, False)], "Host available RAM：全程约 1.89–1.95 TiB，没有持续下滑", y_min=1800, y_max=2050, x_ticks=tick_values, x_formatter=time_label)
    image.save(EVIDENCE / "05_concurrent_formal_resource_timeline.png", optimize=True)


def summarize(
    stage1: list[dict[str, float | int]],
    rlt2: list[dict[str, float | int | None]],
    dsrl: list[dict[str, float | int | None]],
    rlt_resource: list[dict[str, float | datetime]],
    dsrl_resource: list[dict[str, float | datetime]],
) -> dict[str, object]:
    stage1_loss = [float(row["loss"]) for row in stage1]
    stage1_time = [float(row["step_time_s"]) for row in stage1]
    rlt_success = [float(row["train_success"]) for row in rlt2]
    dsrl_success = [float(row["train_success"]) for row in dsrl]
    eval_points = [{"step": int(row["step"]), "success": float(row["eval_success"])} for row in dsrl if row["eval_success"] is not None]
    online = [row for row in dsrl if row["actor_loss"] is not None]
    latest_resource = dsrl_resource[-1]
    rlt_replay = [float(row["global_min_replay_size"]) for row in rlt2]
    rlt_increments = [right - left for left, right in zip(rlt_replay, rlt_replay[1:])]
    dsrl_online_success = [float(row["train_success"]) for row in online]
    dsrl_step_time = [float(row["step_time_s"]) for row in dsrl]
    dsrl_actor_time = [float(row["actor_training_s"]) for row in dsrl if row["actor_training_s"] is not None]
    dsrl_rollout_time = [float(row["generate_rollouts_s"]) for row in dsrl if row["generate_rollouts_s"] is not None]
    steady_cutoff = dsrl_resource[0]["timestamp"] + timedelta(minutes=3)
    steady_dsrl = [row for row in dsrl_resource if row["timestamp"] >= steady_cutoff]
    steady6 = [float(row["gpu6_used_gib"]) for row in steady_dsrl]
    steady7 = [float(row["gpu7_used_gib"]) for row in steady_dsrl]
    return {
        "snapshot_files_downloaded_cst": datetime.now(CST).isoformat(timespec="seconds"),
        "rlt_stage1": {
            "completed_steps": 2000,
            "training_loop_elapsed_s": float(stage1[-1]["elapsed_s"]),
            "wrapper_elapsed_s": 843.0,
            "loss_first100_mean": statistics.mean(stage1_loss[:100]),
            "loss_last100_mean": statistics.mean(stage1_loss[-100:]),
            "loss_final": stage1_loss[-1],
            "median_step_time_s": statistics.median(stage1_time),
        },
        "rlt_stage2": {
            "completed_steps": int(rlt2[-1]["step"]),
            "train_success_mean": statistics.mean(rlt_success),
            "train_success_last5": statistics.mean(rlt_success[-5:]),
            "successful_train_episodes": round(sum(rlt_success) * 8),
            "total_train_episodes": len(rlt_success) * 8,
            "global_total_transitions": 2.0 * float(rlt2[-1]["local_cache_size"]),
            "global_min_replay_size": float(rlt2[-1]["global_min_replay_size"]),
            "median_min_rank_replay_increment_per_cycle": statistics.median(rlt_increments),
            "mean_min_rank_replay_increment_per_cycle": rlt_replay[-1] / len(rlt_replay),
            "estimated_cycle_reaching_warmup_if_resumed": math.ceil(
                int(rlt2[-1]["step"]) + (10000 - rlt_replay[-1]) / (rlt_replay[-1] / len(rlt_replay))
            ),
            "mean_step_time_s": statistics.mean(float(row["step_time_s"]) for row in rlt2),
            "mean_rollout_time_s": statistics.mean(float(row["generate_rollouts_s"]) for row in rlt2),
            "warmup_threshold": 10000,
            "update_step": float(rlt2[-1]["update_step"]),
            "checkpoint_stall_started_utc": STALL_UTC.isoformat(),
            "fixed_eval_points": [{"runner_step": 25, "tensorboard_step": 24, "success": 0.0, "episodes": 20}],
        },
        "dsrl": {
            "completed_steps": int(dsrl[-1]["step"]),
            "first_online_step": int(online[0]["step"]),
            "train_success_mean": statistics.mean(dsrl_success),
            "train_success_last5": statistics.mean(dsrl_success[-5:]),
            "train_success_last20": statistics.mean(dsrl_success[-20:]),
            "warmup_train_success": statistics.mean(dsrl_success[:12]),
            "online_train_success": statistics.mean(dsrl_online_success),
            "successful_train_episodes": round(sum(dsrl_success) * 4),
            "total_train_episodes": len(dsrl_success) * 4,
            "eval_points": eval_points,
            "latest_global_resident_transitions": float(dsrl[-1]["global_resident_transitions"]),
            "cumulative_planned_optimizer_updates": sum(float(row["planned_optimizer_updates"] or 0.0) for row in dsrl),
            "latest_actor_loss": float(dsrl[-1]["actor_loss"]),
            "latest_critic_loss": float(dsrl[-1]["critic_loss"]),
            "latest_alpha": float(dsrl[-1]["alpha"]),
            "latest_actor_grad_norm": float(dsrl[-1]["actor_grad_norm"]),
            "latest_critic_grad_norm": float(dsrl[-1]["critic_grad_norm"]),
            "recent10_mean_step_time_s": statistics.mean(dsrl_step_time[-10:]),
            "recent10_mean_actor_training_s": statistics.mean(dsrl_actor_time[-10:]),
            "recent10_mean_rollout_s": statistics.mean(dsrl_rollout_time[-10:]),
            "latest3_eval_mean": statistics.mean(point["success"] for point in eval_points[-3:]),
        },
        "resources": {
            "rlt_gpu4_peak_gib": max(float(row["gpu4_used_gib"]) for row in rlt_resource),
            "rlt_gpu5_peak_gib": max(float(row["gpu5_used_gib"]) for row in rlt_resource),
            "dsrl_gpu6_peak_gib": max(float(row["gpu6_used_gib"]) for row in dsrl_resource),
            "dsrl_gpu7_peak_gib": max(float(row["gpu7_used_gib"]) for row in dsrl_resource),
            "dsrl_gpu6_steady_median_gib": statistics.median(steady6),
            "dsrl_gpu7_steady_median_gib": statistics.median(steady7),
            "dsrl_gpu6_steady_p95_gib": percentile(steady6, 0.95),
            "dsrl_gpu7_steady_p95_gib": percentile(steady7, 0.95),
            "host_available_min_gib": min(float(row["host_available_gib"]) for row in rlt_resource + dsrl_resource),
            "host_available_latest_gib": float(latest_resource["host_available_gib"]),
            "cgroup_oom_max": max(float(row["cgroup_oom"]) for row in rlt_resource + dsrl_resource),
            "cgroup_oom_kill_max": max(float(row["cgroup_oom_kill"]) for row in rlt_resource + dsrl_resource),
        },
    }


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    stage1 = parse_stage1(EVIDENCE / "rlt_stage1_driver.log")
    rlt2 = parse_metric_tables(EVIDENCE / "rlt_stage2_driver.log", 250, "rlt")
    dsrl = parse_metric_tables(EVIDENCE / "dsrl_driver.log", 200, "dsrl")
    stage1_resource = [
        row
        for row in read_resource(EVIDENCE / "rlt_stage1_resource.csv", (4, 5))
        if row["timestamp"] <= datetime.fromisoformat("2026-08-23T16:44:51+00:00")
    ]
    rlt_resource = read_resource(EVIDENCE / "rlt_stage2_resource.csv", (4, 5))
    dsrl_resource = read_resource(EVIDENCE / "dsrl_resource.csv", (6, 7))

    write_csv(EVIDENCE / "rlt_stage1_metrics_step1_2000.csv", stage1)
    write_csv(EVIDENCE / f"rlt_stage2_metrics_step1_{rlt2[-1]['step']}.csv", rlt2)
    write_csv(
        EVIDENCE / "rlt_stage2_fixed_eval_points.csv",
        [{"runner_step": 25, "tensorboard_step": 24, "success": 0.0, "episodes": 20, "source": "eval/success_once"}],
    )
    write_csv(EVIDENCE / f"dsrl_metrics_step1_{dsrl[-1]['step']}.csv", dsrl)
    summary = summarize(stage1, rlt2, dsrl, rlt_resource, dsrl_resource)
    (EVIDENCE / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    render_stage1(stage1, stage1_resource)
    render_rlt_stage2(rlt2)
    render_dsrl_success(dsrl)
    render_dsrl_optimization(dsrl)
    render_resources(rlt_resource, dsrl_resource)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
