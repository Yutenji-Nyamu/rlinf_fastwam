#!/usr/bin/env python3
"""Build action-aligned DVAC teaching figures for the Shenzhen pi0/Fast-WAM runs.

The script is intentionally offline. It reads the already-downloaded authoritative
CSV bundle plus the seven representative Fast-WAM videos and the 24 representative
pi0 query images. It never reads or changes a live training process.

Figure contract:

* Fast-WAM uses paper-aligned L=5 for task-specific teaching figures. L=3 is kept
  only as a sensitivity/common-cross-policy view.
* pi0 uses L=3 for the main teaching figures and reports L=2/3/4 sensitivity;
  the runtime has only four endpoint estimates, so L=5 is impossible.
* Fast-WAM action slot equals its fresh pre-action video frame only for executed
  h<24. pi0 frames align to query boundaries only.
* All chart limits use the finite min/max with padding. No percentile clipping is
  allowed because the old figures visually clipped important I/R extrema.
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


WORKSPACE = Path(r"C:\Users\86136\Documents\rl")
SOURCE = WORKSPACE / "docs" / "rlinf-shenzhen-pi0-ppo-rlt" / "evidence" / "dvac-analysis-all-four-tasks-20260823"
OUTPUT = WORKSPACE / "docs" / "rlinf-shenzhen-pi0-ppo-rlt" / "evidence" / "dvac-detailed-action-timeline-20260823"
VIDEOS = OUTPUT / "videos"
PI0_IMAGES = OUTPUT / "pi0-query-images"
FIGURES = OUTPUT / "figures"
FRAMES = OUTPUT / "frames"

FONT_REGULAR = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")

WHITE = "#ffffff"
INK = "#172033"
MUTED = "#667085"
GRID = "#d7dde8"
LIGHT = "#f5f7fb"
LIGHT_ALT = "#eef2f7"
PURPLE = "#6941c6"
GREY = "#7a8491"
ORANGE = "#e57200"
BLUE = "#2563b9"
GREEN = "#198a62"
RED = "#d04444"
TEAL = "#168f9c"
BLACK = "#111827"
POST_SUCCESS = "#e4e7ec"

PHASE_COLORS = {
    "approach": "#dbeafe",
    "grasp-contact": "#fed7aa",
    "transport-or-rotate": "#dcfce7",
    "place-release": "#fde68a",
    "verify": "#e9d5ff",
    "UNLABELED": "#e5e7eb",
}


TASK_INFO = {
    "adjust_bottle": {
        "title": "调整瓶子（adjust_bottle）",
        "plain": "双臂接近横放瓶子，建立接触并调整/抬起，最终由 evaluator 判定完成。",
        "warning": "Fast-WAM 本批 16/16 成功，因此只能看成功过程，不能做该模型本任务的成功/失败对照。",
    },
    "move_stapler_pad": {
        "title": "移动订书机到目标垫（move_stapler_pad）",
        "plain": "接近订书机、抓取、搬运/旋转、放到目标区域并验证。",
        "warning": "只有两条代表 episode 有独立 phase 标签；后段未标注保持 UNLABELED。",
    },
    "turn_switch": {
        "title": "拨动开关（turn_switch）",
        "plain": "接近开关并完成精细拨动；代表成功仅 66 actions。",
        "warning": "本任务没有独立 phase CSV；动作阶段文字只作逐帧视觉说明，不冒充总体统计。",
    },
    "pick_diverse_bottles": {
        "title": "抓取多样瓶子（pick_diverse_bottles）",
        "plain": "在随机外观/摆放下接近并抓取瓶子；失败 episode 均运行到 400-action timeout。",
        "warning": "失败只有 4 条；代表 case 的尖峰不能覆盖总体 abs(I) 仍跨 0 的结果。",
    },
}


@dataclass(frozen=True)
class Marker:
    frame: int
    description: str
    terminal: bool = False


@dataclass(frozen=True)
class Case:
    slug: str
    policy: str
    task: str
    episode_uid: str
    outcome: str
    video_name: str | None
    markers: tuple[Marker, ...]
    note: str


CASES: tuple[Case, ...] = (
    Case(
        "pi0_adjust_success",
        "pi0",
        "adjust_bottle",
        "ep000033_reset100100005",
        "success",
        None,
        (
            Marker(0, "q0：初始瓶子"),
            Marker(50, "q1：接触/抓取"),
            Marker(100, "q2：抬起并旋转"),
            Marker(150, "q3：已成功后的输入"),
        ),
        "q3 的 success_before=true，只作描述，不进入 pre-success 主统计。",
    ),
    Case(
        "pi0_adjust_failure",
        "pi0",
        "adjust_bottle",
        "ep000050_reset100100017",
        "failure",
        None,
        (
            Marker(0, "q0：初始瓶子"),
            Marker(50, "q1：抓取尝试"),
            Marker(100, "q2：仍未完成"),
            Marker(150, "q3：末段仍未完成"),
        ),
        "只有 query 输入帧；曲线中的 h 不能声称对应某一物理视频帧。",
    ),
    Case(
        "fastwam_adjust_success",
        "fastwam",
        "adjust_bottle",
        "episode0002_reset1",
        "success",
        "fastwam_adjust_bottle_success_episode0002_reset1.mp4",
        (
            Marker(0, "初始；q0"),
            Marker(16, "接近瓶身"),
            Marker(24, "q1 开始；建立接触"),
            Marker(48, "q2 开始；双臂调整"),
            Marker(72, "q3 开始；继续调整"),
            Marker(96, "q4 开始；末段"),
            Marker(108, "抬起/竖直"),
            Marker(116, "最后执行动作"),
            Marker(117, "成功边界；无 future-h 指标", True),
        ),
        "每个非 terminal 标记都满足 action slot = fresh pre-action video frame。",
    ),
    Case(
        "fastwam_move_success",
        "fastwam",
        "move_stapler_pad",
        "episode0004_reset3",
        "success",
        "fastwam_move_stapler_success_episode0004_reset3.mp4",
        (
            Marker(0, "approach"),
            Marker(24, "grasp-contact 开始"),
            Marker(30, "抓取接触"),
            Marker(48, "transport 开始"),
            Marker(72, "搬运/旋转"),
            Marker(96, "place-release 开始"),
            Marker(118, "放置/释放"),
            Marker(120, "verify 开始"),
            Marker(148, "最后执行动作"),
            Marker(149, "成功边界；无 future-h 指标", True),
        ),
        "phase 标签先于 DVAC 曲线建立；只覆盖这条成功与一条失败 case。",
    ),
    Case(
        "fastwam_move_failure",
        "fastwam",
        "move_stapler_pad",
        "episode0006_reset5",
        "failure",
        "fastwam_move_stapler_failure_episode0006_reset5.mp4",
        (
            Marker(0, "approach"),
            Marker(80, "grasp-contact"),
            Marker(160, "place-release"),
            Marker(216, "q9 开始；后段未标注"),
            Marker(239, "S 与 I 同向峰"),
            Marker(264, "q11 开始；最高 S 区"),
            Marker(288, "q12 开始"),
            Marker(312, "q13 开始"),
            Marker(319, "S 高但 I 抵消"),
            Marker(399, "timeout 前最后动作"),
        ),
        "slot239 与 slot319 是解释 R=S+I 的成对反例；216 以后 phase 保持 UNLABELED。",
    ),
    Case(
        "fastwam_turn_success",
        "fastwam",
        "turn_switch",
        "episode0009_reset8",
        "success",
        "fastwam_turn_switch_success_episode0009_reset8.mp4",
        (
            Marker(0, "初始；q0"),
            Marker(13, "接近开关"),
            Marker(24, "q1 开始"),
            Marker(26, "进入精细操作"),
            Marker(39, "query-wide S 峰"),
            Marker(48, "q2 开始"),
            Marker(52, "拨动后"),
            Marker(65, "最后执行动作"),
            Marker(66, "成功边界；无 future-h 指标", True),
        ),
        "高 S 在本任务可能表示真正进入有效操作阶段，而非失败风险。",
    ),
    Case(
        "fastwam_turn_failure",
        "fastwam",
        "turn_switch",
        "episode0001_reset0",
        "failure",
        "fastwam_turn_switch_failure_episode0001_reset0.mp4",
        (
            Marker(0, "初始；q0"),
            Marker(24, "q1 开始"),
            Marker(39, "早期高 S 区"),
            Marker(48, "q2 开始"),
            Marker(80, "未形成有效操作"),
            Marker(160, "长期停滞"),
            Marker(239, "仍未完成"),
            Marker(319, "仍未完成"),
            Marker(360, "timeout 后段"),
            Marker(399, "最后动作"),
        ),
        "没有正式 phase 标签；只描述画面和曲线，不把视觉观察扩展成总体因果结论。",
    ),
    Case(
        "fastwam_pick_success",
        "fastwam",
        "pick_diverse_bottles",
        "episode0004_reset3",
        "success",
        "fastwam_pick_bottles_success_episode0004_reset3.mp4",
        (
            Marker(0, "初始瓶子"),
            Marker(21, "双臂接近"),
            Marker(24, "q1 开始"),
            Marker(42, "建立接触"),
            Marker(48, "q2 开始"),
            Marker(63, "抓取/稳定"),
            Marker(72, "q3 开始"),
            Marker(84, "抬起"),
            Marker(96, "q4 开始"),
            Marker(105, "最后执行动作"),
            Marker(106, "成功边界；无 future-h 指标", True),
        ),
        "本任务没有独立 phase 标签；帧说明只用于理解当前代表 case。",
    ),
    Case(
        "fastwam_pick_failure",
        "fastwam",
        "pick_diverse_bottles",
        "episode0005_reset4",
        "failure",
        "fastwam_pick_bottles_failure_episode0005_reset4.mp4",
        (
            Marker(0, "初始瓶子"),
            Marker(80, "抓取尝试"),
            Marker(160, "仍未完成"),
            Marker(168, "q7：持续抬升起点"),
            Marker(216, "q9：中后段"),
            Marker(288, "q12：query-wide S 高区"),
            Marker(319, "q13 内动作"),
            Marker(360, "q15：后段 S 高区"),
            Marker(384, "q16 开始"),
            Marker(399, "最后动作；timeout"),
        ),
        "代表失败后段 S 持续抬高；全体失败的 abs(I) 区间仍跨 0。",
    ),
)


METRICS = {
    "raw": {
        "title": "裸 DVAC：V_L 的对数视图 y = ln(V_L + 10^-12)",
        "plain": "回答：去噪末段对这个 future action 还改口多少？图画 y 而非直接画极小的 V；峰值位置完全相同。",
        "series": (("y_ln", "当前 y", PURPLE), ("b_position", "同 h 的位置基线 b_h", GREY)),
        "unit": "natural-log unit",
    },
    "r": {
        "title": "去位置后的 raw residual：r = y - b_h",
        "plain": "回答：同样位于第 h 格，这一次比该位置通常水平高/低多少；仍是 raw log 单位。",
        "series": (("r_raw", "r", ORANGE),),
        "unit": "log unit",
    },
    "R": {
        "title": "标准化 residual：R = r / s_h",
        "plain": "回答：相对同一 h 自己的典型波动，这次高/低了多少个 robust scale；不是概率或风险阈值。",
        "series": (("R_std", "R", BLUE),),
        "unit": "robust-scale unit",
    },
    "S": {
        "title": "query-wide 信号：S(q) = mean_h R(q,h)",
        "plain": "回答：当前 query 的整条 future chunk 是否一起抬高/降低；更像 state/task-stage 的共同效应。",
        "series": (("S_std", "S", GREEN),),
        "unit": "robust-scale unit",
    },
    "I": {
        "title": "query 内局部信号：I(q,h) = R(q,h) - S(q)",
        "plain": "回答：扣掉当前 query 整体升降后，这个具体 future action 还额外突出多少；每个 query 内均值为 0。",
        "series": (("I_std", "I", ORANGE),),
        "unit": "robust-scale unit",
    },
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size=size)


def wrap_text(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for char in paragraph:
            candidate = current + char
            if len(candidate.encode("utf-8")) > width and current:
                lines.append(current)
                current = char
            else:
                current = candidate
        if current:
            lines.append(current)
    return lines


def draw_text_block(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    width_chars: int,
    size: int,
    fill: str = INK,
    bold: bool = False,
    spacing: int = 6,
) -> int:
    y = xy[1]
    fnt = font(size, bold)
    for line in wrap_text(text, width_chars):
        draw.text((xy[0], y), line, font=fnt, fill=fill)
        y += size + spacing
    return y


def dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    fill: str,
    width: int = 2,
    dash: int = 8,
    gap: int = 6,
) -> None:
    x1, y1 = start
    x2, y2 = end
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    pos = 0.0
    while pos < length:
        stop = min(pos + dash, length)
        draw.line((x1 + dx * pos, y1 + dy * pos, x1 + dx * stop, y1 + dy * stop), fill=fill, width=width)
        pos += dash + gap


def safe_minmax(values: Iterable[np.ndarray], *, include_zero: bool = False) -> tuple[float, float]:
    finite: list[np.ndarray] = []
    for value in values:
        array = np.asarray(value, dtype=float)
        array = array[np.isfinite(array)]
        if array.size:
            finite.append(array)
    if not finite:
        return -1.0, 1.0
    merged = np.concatenate(finite)
    low = float(merged.min())
    high = float(merged.max())
    if include_zero:
        low = min(low, 0.0)
        high = max(high, 0.0)
    if math.isclose(low, high):
        pad = max(abs(low) * 0.1, 1.0)
    else:
        pad = (high - low) * 0.09
    return low - pad, high + pad


def map_value(value: float, low: float, high: float, p_low: float, p_high: float) -> float:
    if math.isclose(low, high):
        return (p_low + p_high) / 2
    return p_low + (value - low) / (high - low) * (p_high - p_low)


def read_video_frame(path: Path, index: int) -> Image.Image:
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame_bgr = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {index} from {path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    finally:
        capture.release()


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.fit(image.convert("RGB"), size, method=Image.Resampling.LANCZOS)


def episode_suffix(frame: pd.DataFrame, episode_uid: str) -> pd.DataFrame:
    if "episode_uid" in frame.columns:
        selected = frame[frame["episode_uid"].astype(str) == episode_uid]
        if len(selected):
            return selected.copy()
    return frame[frame["episode_key"].astype(str).str.endswith(episode_uid)].copy()


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    query = pd.read_csv(SOURCE / "query_metrics.csv", low_memory=False)
    horizon = pd.read_csv(SOURCE / "query_horizon.csv", low_memory=False)
    episode = pd.read_csv(SOURCE / "episode_metrics.csv", low_memory=False)
    l_sensitivity = pd.read_csv(SOURCE / "L_sensitivity.csv", low_memory=False)
    outcome = pd.read_csv(SOURCE / "outcome_summary.csv", low_memory=False)
    return query, horizon, episode, l_sensitivity, outcome


def main_l(policy: str) -> int:
    return 5 if policy == "fastwam" else 3


def prepare_case_data(case: Case, horizon: pd.DataFrame, episode: pd.DataFrame) -> pd.DataFrame:
    selected = episode_suffix(horizon, case.episode_uid)
    selected = selected[(selected["policy"] == case.policy) & (selected["task"] == case.task)].copy()
    selected = selected[(selected["L"] == main_l(case.policy)) & truthy(selected["executed_flag"])].copy()
    selected["action_slot"] = pd.to_numeric(selected["action_slot"], errors="coerce")
    selected = selected.dropna(subset=["action_slot"]).sort_values("action_slot")
    if selected.empty:
        raise ValueError(f"No case rows for {case.slug}")
    erow = episode_suffix(episode, case.episode_uid)
    erow = erow[(erow["policy"] == case.policy) & (erow["task"] == case.task)].copy()
    if erow.empty:
        raise ValueError(f"No episode row for {case.slug}")
    selected["video_path_local"] = str(VIDEOS / case.video_name) if case.video_name else ""
    selected["total_action_slots"] = int(float(erow.iloc[0]["total_action_slots"]))
    return selected


def truthy(series: pd.Series) -> pd.Series:
    """Parse bool-like CSV columns without treating the string 'False' as true."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def marker_letter(index: int) -> str:
    return chr(ord("A") + index)


def local_frame(case: Case, marker: Marker) -> Image.Image:
    """Return one exact Fast-WAM frame or one pi0 three-camera query mosaic."""
    if case.policy == "fastwam":
        assert case.video_name is not None
        return read_video_frame(VIDEOS / case.video_name, marker.frame)

    query_index = int(marker.frame // 50)
    head = Image.open(PI0_IMAGES / f"{case.episode_uid}__q{query_index:02d}__head.png").convert("RGB")
    left = Image.open(PI0_IMAGES / f"{case.episode_uid}__q{query_index:02d}__left.png").convert("RGB")
    right = Image.open(PI0_IMAGES / f"{case.episode_uid}__q{query_index:02d}__right.png").convert("RGB")
    mosaic = Image.new("RGB", (520, 360), WHITE)
    mosaic.paste(fit_image(head, (520, 230)), (0, 0))
    mosaic.paste(fit_image(left, (256, 124)), (0, 236))
    mosaic.paste(fit_image(right, (256, 124)), (264, 236))
    return mosaic


def phase_ranges(data: pd.DataFrame) -> list[tuple[int, int, str]]:
    if "phase_task" not in data.columns:
        return []
    rows = data[["action_slot", "phase_task"]].drop_duplicates("action_slot").sort_values("action_slot")
    if rows.empty:
        return []
    rows["phase_task"] = rows["phase_task"].fillna("UNLABELED").astype(str)
    result: list[tuple[int, int, str]] = []
    start = int(rows.iloc[0]["action_slot"])
    previous = start
    phase = str(rows.iloc[0]["phase_task"])
    for row in rows.iloc[1:].itertuples(index=False):
        slot = int(row.action_slot)
        current = str(row.phase_task)
        if current != phase or slot != previous + 1:
            result.append((start, previous + 1, phase))
            start = slot
            phase = current
        previous = slot
    result.append((start, previous + 1, phase))
    return result


def query_ranges(data: pd.DataFrame) -> list[tuple[int, int, int]]:
    result: list[tuple[int, int, int]] = []
    for query_idx, group in data.groupby("query_idx", sort=True):
        slots = pd.to_numeric(group["action_slot"], errors="coerce").dropna().astype(int)
        if len(slots):
            result.append((int(slots.min()), int(slots.max()) + 1, int(query_idx)))
    return result


def marker_row(data: pd.DataFrame, frame: int) -> pd.Series | None:
    rows = data[pd.to_numeric(data["action_slot"], errors="coerce") == frame]
    return None if rows.empty else rows.iloc[0]


def render_case_metric(case: Case, metric_key: str, data: pd.DataFrame) -> dict[str, object]:
    """Render one signal only, with exact frame cards and letter-matched curve points."""
    meta = METRICS[metric_key]
    width = 3360
    margin = 90
    frame_top = 250
    frame_h = 350
    chart_top = 720
    chart_bottom = 1400
    table_top = 1455
    height = 1870
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)

    outcome_color = GREEN if case.outcome == "success" else RED
    outcome_cn = "成功" if case.outcome == "success" else "失败"
    draw.text((margin, 55), f"{TASK_INFO[case.task]['title']}｜{outcome_cn}代表 case｜{meta['title']}", font=font(48, True), fill=INK)
    draw.text((margin, 125), f"{case.policy} · {case.episode_uid} · 主口径 L={main_l(case.policy)}", font=font(28, True), fill=outcome_color)
    align_text = (
        "Fast-WAM：横轴 action slot 与 fresh pre-action 视频 frame 精确相同；只画实际执行的 h<24。"
        if case.policy == "fastwam"
        else "π0：上方仅是每次 policy query 的三相机输入；不能把 chunk 内 h 当成物理视频帧。"
    )
    draw.text((margin, 173), align_text, font=font(23), fill=MUTED)

    card_gap = 16
    card_w = int((width - 2 * margin - card_gap * (len(case.markers) - 1)) / len(case.markers))
    card_w = max(245, min(card_w, 380))
    strip_w = len(case.markers) * card_w + (len(case.markers) - 1) * card_gap
    start_x = max(margin, (width - strip_w) // 2)
    saved_frames: list[str] = []
    FRAMES.joinpath(case.slug).mkdir(parents=True, exist_ok=True)
    for idx, marker in enumerate(case.markers):
        x = start_x + idx * (card_w + card_gap)
        card = local_frame(case, marker)
        frame_img = fit_image(card, (card_w, 230))
        image.paste(frame_img, (x, frame_top))
        border = GREEN if marker.terminal else GRID
        draw.rounded_rectangle((x, frame_top, x + card_w, frame_top + frame_h), radius=12, outline=border, width=3, fill=None)
        letter = marker_letter(idx)
        draw.ellipse((x + 10, frame_top + 10, x + 56, frame_top + 56), fill=outcome_color)
        draw.text((x + 23, frame_top + 15), letter, font=font(26, True), fill=WHITE)
        axis_name = "frame" if case.policy == "fastwam" else "query start"
        draw.text((x + 10, frame_top + 244), f"{axis_name} {marker.frame}", font=font(19, True), fill=INK)
        draw_text_block(draw, (x + 10, frame_top + 274), marker.description, width_chars=max(18, card_w // 13), size=18, fill=MUTED, spacing=3)
        frame_path = FRAMES / case.slug / f"{letter}_{marker.frame:04d}.png"
        card.save(frame_path)
        saved_frames.append(str(frame_path.relative_to(OUTPUT)).replace("\\", "/"))

    x0, x1 = margin + 125, width - margin
    y0, y1 = chart_top, chart_bottom
    draw.rounded_rectangle((margin, chart_top - 80, width - margin, chart_bottom + 10), radius=20, fill=LIGHT, outline=GRID, width=2)
    draw.text((margin + 28, chart_top - 61), meta["plain"], font=font(24, True), fill=INK)

    x_values = pd.to_numeric(data["action_slot"], errors="coerce").to_numpy(dtype=float)
    x_min = 0.0
    x_max = float(max(int(data["total_action_slots"].iloc[0]) - 1, 1))
    series_arrays = [pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float) for column, _, _ in meta["series"]]
    y_min, y_max = safe_minmax(series_arrays, include_zero=metric_key != "raw")

    for start, end, qidx in query_ranges(data):
        px0 = map_value(start, x_min, x_max, x0, x1)
        px1 = map_value(min(end, x_max), x_min, x_max, x0, x1)
        fill = POST_SUCCESS if case.policy == "pi0" and case.outcome == "success" and qidx == 3 else ("#ffffff" if qidx % 2 == 0 else LIGHT_ALT)
        draw.rectangle((px0, y0, px1, y1), fill=fill)
        qlabel = f"q{qidx} · post-success" if case.policy == "pi0" and case.outcome == "success" and qidx == 3 else f"q{qidx}"
        draw.text((px0 + 8, y0 + 8), qlabel, font=font(17, True), fill="#667085" if "post" in qlabel else "#98a2b3")

    phases = phase_ranges(data)
    if phases:
        for start, end, phase in phases:
            px0 = map_value(start, x_min, x_max, x0, x1)
            px1 = map_value(min(end, x_max), x_min, x_max, x0, x1)
            draw.rectangle((px0, y0, px1, y0 + 34), fill=PHASE_COLORS.get(phase, PHASE_COLORS["UNLABELED"]))
            if px1 - px0 > 90:
                draw.text((px0 + 5, y0 + 5), phase, font=font(14, True), fill=INK)

    for tick in np.linspace(y_min, y_max, 6):
        py = map_value(float(tick), y_min, y_max, y1, y0)
        draw.line((x0, py, x1, py), fill=GRID, width=2)
        draw.text((margin + 5, py - 12), f"{tick:+.2f}", font=font(18), fill=MUTED)
    for tick in np.linspace(x_min, x_max, 9):
        px = map_value(float(tick), x_min, x_max, x0, x1)
        draw.line((px, y0, px, y1), fill=GRID, width=1)
        draw.text((px - 18, y1 + 16), f"{int(round(tick))}", font=font(17), fill=MUTED)
    if metric_key != "raw" and y_min <= 0 <= y_max:
        py0 = map_value(0.0, y_min, y_max, y1, y0)
        draw.line((x0, py0, x1, py0), fill="#68737d", width=3)

    legend_x = x0 + 20
    for column, label, color in meta["series"]:
        vals = pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float)
        points = []
        for xval, value in zip(x_values, vals):
            if math.isfinite(xval) and math.isfinite(value):
                points.append((map_value(xval, x_min, x_max, x0, x1), map_value(value, y_min, y_max, y1, y0)))
        if len(points) > 1:
            draw.line(points, fill=color, width=5, joint="curve")
        draw.line((legend_x, chart_top - 25, legend_x + 70, chart_top - 25), fill=color, width=6)
        draw.text((legend_x + 82, chart_top - 42), label, font=font(22, True), fill=color)
        legend_x += 380

    marker_records: list[dict[str, object]] = []
    for idx, marker in enumerate(case.markers):
        letter = marker_letter(idx)
        record: dict[str, object] = {
            "case": case.slug,
            "policy": case.policy,
            "task": case.task,
            "outcome": case.outcome,
            "marker": letter,
            "action_or_query_slot": marker.frame,
            "description": marker.description,
            "terminal_no_metric": marker.terminal,
        }
        row = marker_row(data, marker.frame)
        if not marker.terminal and row is not None:
            px = map_value(marker.frame, x_min, x_max, x0, x1)
            first_column = meta["series"][0][0]
            value = float(row[first_column])
            py = map_value(value, y_min, y_max, y1, y0)
            dashed_line(draw, (px, frame_top + frame_h), (px, y1), fill="#98a2b3", width=2, dash=9, gap=7)
            draw.ellipse((px - 14, py - 14, px + 14, py + 14), fill=outcome_color, outline=WHITE, width=3)
            draw.text((px - 10, py - 45), letter, font=font(23, True), fill=outcome_color)
            for column in ("V", "y_ln", "b_position", "mad_scale", "r_raw", "R_std", "S_std", "I_std"):
                if column in row.index:
                    record[column] = float(row[column])
        marker_records.append(record)

    draw.text((x0, y1 + 50), "episode action slot / fresh pre-action video frame" if case.policy == "fastwam" else "episode action slot；每 50 个 slot 仅标一个 query 输入", font=font(21, True), fill=MUTED)
    draw.text((margin, table_top), "标号读法：上方画面与曲线同字母精确对应；terminal 成功边界没有 future-h 指标。", font=font(24, True), fill=INK)
    draw.text((margin, table_top + 42), case.note, font=font(21), fill=MUTED)

    col_w = (width - 2 * margin) // min(len(case.markers), 5)
    for idx, record in enumerate(marker_records):
        row_idx = idx // 5
        col_idx = idx % 5
        bx = margin + col_idx * col_w
        by = table_top + 95 + row_idx * 145
        draw.rounded_rectangle((bx, by, bx + col_w - 15, by + 125), radius=12, fill=LIGHT, outline=GRID, width=2)
        draw.text((bx + 12, by + 10), f"{record['marker']} · slot {record['action_or_query_slot']}", font=font(21, True), fill=outcome_color)
        if record["terminal_no_metric"]:
            value_text = "terminal 边界；无指标"
        elif metric_key == "raw":
            value_text = f"y={record.get('y_ln', float('nan')):+.3f}；b={record.get('b_position', float('nan')):+.3f}"
        else:
            column = meta["series"][0][0]
            value_text = f"{meta['series'][0][1]}={record.get(column, float('nan')):+.3f}"
        draw.text((bx + 12, by + 47), value_text, font=font(19, True), fill=INK)
        draw_text_block(draw, (bx + 12, by + 78), str(record["description"]), width_chars=max(22, col_w // 13), size=17, fill=MUTED, spacing=2)

    metric_file = {
        "raw": "raw_dvac",
        "r": "raw_residual",
        "R": "standardized_residual",
        "S": "query_wide_S",
        "I": "local_I",
    }[metric_key]
    # Windows is case-insensitive, so never distinguish raw r from standardized R
    # by letter case alone.
    output_name = f"case_{case.slug}__{metric_file}.png"
    output_path = FIGURES / output_name
    image.save(output_path, optimize=True)
    return {
        "figure": str(output_path.relative_to(OUTPUT)).replace("\\", "/"),
        "kind": "case_metric",
        "policy": case.policy,
        "task": case.task,
        "case": case.slug,
        "outcome": case.outcome,
        "metric": metric_key,
        "main_L": main_l(case.policy),
        "frames": saved_frames,
        "marker_records": marker_records,
    }


def signal_dictionary_figure() -> str:
    width, height = 2600, 1520
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    draw.text((80, 55), "DVAC 分解不是“7个神秘指标”：它是一条逐层记账链", font=font(48, True), fill=INK)
    draw.text((80, 122), "先记录去噪末段改口，再去固定位置、按位置尺度标准化，最后拆成 query 整体与局部 future-action。", font=font(27), fill=MUTED)
    cards = [
        ("1  V_L(q,h)", "裸 DVAC", "最后 L 个 clean endpoint 在14个动作维度上的 population variance 之和。越大=末段去噪仍在改口；不是失败概率。", PURPLE),
        ("2  y(q,h)", "可视化尺度", "y = ln(V_L + 10^-12)。ln 压缩跨数量级数值；10^-12 只防止 V=0 时 ln(0)，不会制造峰。", PURPLE),
        ("3  b_h", "位置中心", "同任务、同 future 位置 h 的 y 中位数。它回答“第 h 格通常多高”，是参照，不是当前 episode 信号。", GREY),
        ("4  s_h", "位置尺度", "s_h = 1.4826×MAD，并设最小 10^-6。它回答“第 h 格自己通常波动多大”。", GREY),
        ("5  r(q,h)", "raw residual", "r = y-b_h。只去固定位置中心，仍保留 raw log 单位；不同 h 的正常波动还不可直接比较。", ORANGE),
        ("6  R(q,h)", "标准化 residual", "R = r/s_h。表示相对这个 h 的典型波动高低了多少 robust scale；R 仍不是概率。", BLUE),
        ("7  S(q)", "query-wide", "S = mean_h R。整条 future chunk 一起升降，更容易携带当前 state / task-stage 的共同效应。", GREEN),
        ("8  I(q,h)", "local interaction", "I = R-S。同一个 query 内某个 future action 额外突出多少；每个 query 内 mean_h I = 0。", ORANGE),
    ]
    card_w, card_h = 580, 520
    for idx, (symbol, subtitle, body, color) in enumerate(cards):
        row, col = divmod(idx, 4)
        x = 80 + col * 630
        y = 230 + row * 585
        draw.rounded_rectangle((x, y, x + card_w, y + card_h), radius=22, fill=LIGHT, outline=color, width=4)
        draw.text((x + 28, y + 28), symbol, font=font(32, True), fill=color)
        draw.text((x + 28, y + 82), subtitle, font=font(25, True), fill=INK)
        draw_text_block(draw, (x + 28, y + 145), body, width_chars=47, size=23, fill=MUTED, spacing=9)
    draw.rounded_rectangle((80, 1400, 2520, 1490), radius=18, fill="#fff7ed", outline=ORANGE, width=3)
    draw.text((105, 1427), "两条严格恒等式：y = b_h + r；R = S + I。不能把标准化的 S/I 直接加回 raw y。", font=font(27, True), fill=INK)
    path = FIGURES / "reference_01_signal_dictionary.png"
    image.save(path, optimize=True)
    return str(path.relative_to(OUTPUT)).replace("\\", "/")


def denoising_window_figure() -> str:
    width, height = 2500, 1120
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    draw.text((80, 55), "L 到底看哪几步：看 clean endpoint estimates，不是执行多少个 action", font=font(46, True), fill=INK)
    draw.text((80, 120), "z_i = x_i - t_i v_i；DVAC 对选中的 z_i 做 population variance。最终动作 chunk x_M 不是额外的第六个样本。", font=font(25), fill=MUTED)

    def row(y: int, title: str, m: int, selected: set[int], color: str, note: str) -> None:
        draw.text((95, y), title, font=font(31, True), fill=INK)
        start_x, box_w, gap = 420, 145, 30
        for i in range(m):
            x = start_x + i * (box_w + gap)
            fill = color if i in selected else LIGHT_ALT
            text_fill = WHITE if i in selected else MUTED
            draw.rounded_rectangle((x, y - 12, x + box_w, y + 100), radius=18, fill=fill, outline=GRID, width=2)
            draw.text((x + 45, y + 18), f"z{i}", font=font(28, True), fill=text_fill)
        draw_text_block(draw, (420, y + 135), note, width_chars=150, size=23, fill=MUTED, spacing=5)

    row(260, "Fast-WAM  M=10", 10, set(range(5, 10)), PURPLE, "论文默认 L=5：使用 z5,z6,z7,z8,z9。当前教学主图采用这个口径；旧 L=3 仅留作敏感性对照。")
    row(650, "π0  M=4", 4, {1, 2, 3}, BLUE, "π0 只有 z0..z3，因此不能伪造 L=5。主图用 L=3（z1,z2,z3），并报告 L=2/3/4 的敏感性。")
    path = FIGURES / "reference_02_denoising_tail_window.png"
    image.save(path, optimize=True)
    return str(path.relative_to(OUTPUT)).replace("\\", "/")


def alignment_figure() -> str:
    width, height = 2600, 1420
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    draw.text((80, 55), "三个容易混淆的轴：query q、future 位置 h、episode action slot", font=font(48, True), fill=INK)
    draw.text((80, 122), "去噪步 i 是第四个轴；它只用于计算每个 (q,h) 的 V_L，不等于 action 时间。", font=font(26), fill=MUTED)

    draw.text((100, 235), "Fast-WAM：H=32，最多执行 C=24", font=font(34, True), fill=INK)
    x0, x1 = 130, 2470
    cell = (x1 - x0) / 32
    for h in range(32):
        x = x0 + h * cell
        fill = GREEN if h < 24 else "#e5e7eb"
        draw.rectangle((x, 310, x + cell - 3, 405), fill=fill, outline=WHITE, width=1)
        if h % 4 == 0:
            draw.text((x + 6, 335), str(h), font=font(18, True), fill=WHITE if h < 24 else MUTED)
    draw.text((130, 435), "h=0..23：实际执行，action slot = fresh pre-action video frame", font=font(25, True), fill=GREEN)
    draw.text((1690, 435), "h=24..31：预测但未执行；没有对应视频帧", font=font(24, True), fill=MUTED)

    draw.text((100, 550), "连续 query 示例：每个色块是一条新预测 chunk", font=font(32, True), fill=INK)
    for q in range(4):
        start = 130 + q * 565
        draw.rounded_rectangle((start, 625, start + 520, 745), radius=14, fill=LIGHT if q % 2 == 0 else LIGHT_ALT, outline=GRID, width=2)
        draw.text((start + 20, 650), f"q{q}", font=font(28, True), fill=INK)
        draw.text((start + 95, 650), f"slots {q*24}..{q*24+23}", font=font(25), fill=MUTED)
    draw.text((130, 790), "因此 Fast-WAM 可以把 action-level R/I 精确叠到视频帧；S 在同一 query 的24个执行 action上保持不变。", font=font(26, True), fill=INK)

    draw.text((100, 915), "π0：H=C=50，但保存的是 query-boundary 三相机输入", font=font(34, True), fill=INK)
    for q in range(4):
        start = 130 + q * 565
        draw.rounded_rectangle((start, 990, start + 520, 1110), radius=14, fill=LIGHT if q % 2 == 0 else LIGHT_ALT, outline=GRID, width=2)
        draw.text((start + 20, 1015), f"q{q}", font=font(28, True), fill=INK)
        draw.text((start + 95, 1015), f"query input at slot {q*50}", font=font(24), fill=MUTED)
    draw.rounded_rectangle((100, 1190, 2500, 1340), radius=20, fill="#fff1f2", outline=RED, width=3)
    draw_text_block(draw, (135, 1225), "关键限制：π0 的 h=17 只是“当前 query 预测的第17个 future action”，不是 episode 视频第17帧。现有材料只能把 S 等 query-level量放到四个 query 边界，不能补造逐action视觉对应。", width_chars=165, size=25, fill=INK, bold=True, spacing=7)
    path = FIGURES / "reference_03_axis_and_video_alignment.png"
    image.save(path, optimize=True)
    return str(path.relative_to(OUTPUT)).replace("\\", "/")


def position_reference_figures(horizon: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    groups = horizon[["policy", "task"]].drop_duplicates().sort_values(["policy", "task"])
    for policy, task in groups.itertuples(index=False):
        l_value = main_l(str(policy))
        data = horizon[(horizon["policy"] == policy) & (horizon["task"] == task) & (horizon["L"] == l_value)].copy()
        base = data.groupby("h", as_index=False).agg(b_position=("b_position", "first"), mad_scale=("mad_scale", "first"))
        width, height = 2400, 1180
        image = Image.new("RGB", (width, height), WHITE)
        draw = ImageDraw.Draw(image)
        draw.text((80, 55), f"固定 future 位置参照｜{policy} · {TASK_INFO[str(task)]['title']} · L={l_value}", font=font(44, True), fill=INK)
        draw.text((80, 120), "b_h 是该 h 的 y 中位数；s_h 是该 h 的 1.4826×MAD。二者都由同组 pre-success queries 离线估计。", font=font(25), fill=MUTED)

        def panel(top: int, column: str, title: str, color: str, include_zero: bool) -> None:
            left, right, bottom = 180, 2310, top + 360
            vals = pd.to_numeric(base[column], errors="coerce").to_numpy(float)
            xs = pd.to_numeric(base["h"], errors="coerce").to_numpy(float)
            ymin, ymax = safe_minmax([vals], include_zero=include_zero)
            draw.text((90, top - 52), title, font=font(29, True), fill=color)
            for tick in np.linspace(ymin, ymax, 5):
                py = map_value(float(tick), ymin, ymax, bottom, top)
                draw.line((left, py, right, py), fill=GRID, width=2)
                draw.text((95, py - 11), f"{tick:+.2f}", font=font(18), fill=MUTED)
            points = [(map_value(x, xs.min(), xs.max(), left, right), map_value(v, ymin, ymax, bottom, top)) for x, v in zip(xs, vals)]
            draw.line(points, fill=color, width=6, joint="curve")
            for x, v in points:
                draw.ellipse((x - 5, v - 5, x + 5, v + 5), fill=color)
            for tick in range(0, int(xs.max()) + 1, max(1, int(xs.max() // 8))):
                px = map_value(tick, xs.min(), xs.max(), left, right)
                draw.text((px - 12, bottom + 14), str(tick), font=font(18), fill=MUTED)

        panel(260, "b_position", "位置中心 b_h：第 h 格通常有多高（raw log单位）", GREY, False)
        panel(760, "mad_scale", "位置尺度 s_h：第 h 格自己通常波动多大（用于 r/s_h）", TEAL, True)
        draw.text((970, 1130), "future position h", font=font(22, True), fill=MUTED)
        path = FIGURES / f"position_reference_{policy}_{task}.png"
        image.save(path, optimize=True)
        records.append({"figure": str(path.relative_to(OUTPUT)).replace("\\", "/"), "kind": "position_reference", "policy": policy, "task": task, "main_L": l_value})
    return records


def episode_metric_table(horizon: pd.DataFrame) -> pd.DataFrame:
    selected = horizon[truthy(horizon["baseline_eligible"])].copy()
    selected["abs_R"] = pd.to_numeric(selected["R_std"], errors="coerce").abs()
    selected["abs_I"] = pd.to_numeric(selected["I_std"], errors="coerce").abs()
    per_query = selected.groupby(["policy", "task", "episode_key", "success", "L", "query_idx"], as_index=False).agg(
        S=("S_std", "first"), abs_R=("abs_R", "mean"), abs_I=("abs_I", "mean")
    )
    melted = per_query.melt(
        id_vars=["policy", "task", "episode_key", "success", "L", "query_idx"],
        value_vars=["S", "abs_R", "abs_I"],
        var_name="metric", value_name="value",
    )
    return melted.groupby(["policy", "task", "episode_key", "success", "L", "metric"], as_index=False)["value"].mean()


def spearman_pair(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 3 or left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return float("nan")
    return float(left.rank(method="average").corr(right.rank(method="average"), method="pearson"))


def sensitivity_tables(horizon: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_metric = episode_metric_table(horizon)
    rank_rows: list[dict[str, object]] = []
    outcome_rows: list[dict[str, object]] = []
    for (policy, task, metric), group in episode_metric.groupby(["policy", "task", "metric"], sort=True):
        levels = sorted(group["L"].unique().tolist())
        pairs: list[tuple[int, int]] = []
        if policy == "fastwam" and 3 in levels and 5 in levels:
            pairs = [(3, 5)]
        elif policy == "pi0":
            pairs = [(int(a), int(b)) for a, b in zip(levels[:-1], levels[1:])]
        for left_l, right_l in pairs:
            wide = group[group["L"].isin([left_l, right_l])].pivot_table(index=["episode_key", "success"], columns="L", values="value", aggfunc="first").dropna()
            for scope, mask in (
                ("all", np.ones(len(wide), dtype=bool)),
                ("success", wide.index.get_level_values("success").astype(bool)),
                ("failure", ~wide.index.get_level_values("success").astype(bool)),
            ):
                scoped = wide.loc[mask]
                rank_rows.append({
                    "policy": policy, "task": task, "metric": metric, "scope": scope,
                    "L_left": left_l, "L_right": right_l, "episodes_paired": len(scoped),
                    "rho_spearman": spearman_pair(scoped[left_l], scoped[right_l]),
                    "aggregation_unit": "episode_equal_mean_of_query_metrics",
                })
        for l_value, lgroup in group.groupby("L"):
            successes = lgroup[truthy(lgroup["success"])]["value"].dropna().to_numpy(float)
            failures = lgroup[~truthy(lgroup["success"])]["value"].dropna().to_numpy(float)
            if len(successes) and len(failures):
                rng = np.random.default_rng(zlib.crc32(f"{policy}|{task}|{metric}|{l_value}".encode("utf-8")))
                draws = np.empty(10000, dtype=float)
                for draw_idx in range(len(draws)):
                    draws[draw_idx] = float(rng.choice(successes, len(successes), replace=True).mean() - rng.choice(failures, len(failures), replace=True).mean())
                ci_low, ci_high = np.quantile(draws, [0.025, 0.975]).tolist()
            else:
                ci_low, ci_high = float("nan"), float("nan")
            outcome_rows.append({
                "policy": policy, "task": task, "metric": metric, "L": int(l_value),
                "success_episodes": len(successes), "failure_episodes": len(failures),
                "success_mean": float(successes.mean()) if len(successes) else float("nan"),
                "success_sd": float(successes.std(ddof=1)) if len(successes) > 1 else float("nan"),
                "failure_mean": float(failures.mean()) if len(failures) else float("nan"),
                "failure_sd": float(failures.std(ddof=1)) if len(failures) > 1 else float("nan"),
                "success_minus_failure": float(successes.mean() - failures.mean()) if len(successes) and len(failures) else float("nan"),
                "bootstrap_ci_low": ci_low, "bootstrap_ci_high": ci_high,
                "aggregation_unit": "episode_equal_mean_of_full_H_query_metrics",
            })
    return pd.DataFrame(rank_rows), pd.DataFrame(outcome_rows)


def sensitivity_figure(rank_table: pd.DataFrame, outcome_table: pd.DataFrame) -> str:
    width, height = 2800, 1780
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    draw.text((80, 55), "去噪 tail 长度敏感性：Fast-WAM 的 L=3 与论文默认 L=5 是否讲同一件事", font=font(46, True), fill=INK)
    draw.text((80, 120), "排名相关按 episode 等权；outcome 差也先压成一episode一值。高相关只表示排序稳，不表示信号就是校准风险。", font=font(25), fill=MUTED)
    fw = rank_table[(rank_table["policy"] == "fastwam") & (rank_table["scope"] == "all")].copy()
    tasks = ["adjust_bottle", "move_stapler_pad", "turn_switch", "pick_diverse_bottles"]
    metrics = ["S", "abs_R", "abs_I"]
    x0, y0, cw, ch = 740, 280, 470, 170
    draw.text((85, 290), "episode Spearman ρ", font=font(30, True), fill=INK)
    for j, task in enumerate(tasks):
        draw.text((x0 + j * cw + 15, y0 - 55), task.replace("_", "\n"), font=font(22, True), fill=INK)
    for i, metric in enumerate(metrics):
        draw.text((120, y0 + i * ch + 55), metric, font=font(28, True), fill=INK)
        for j, task in enumerate(tasks):
            row = fw[(fw["task"] == task) & (fw["metric"] == metric)]
            value = float(row.iloc[0]["rho_spearman"]) if len(row) else float("nan")
            t = 0.5 if not math.isfinite(value) else max(0.0, min(1.0, (value + 1) / 2))
            color = tuple(int((1 - t) * a + t * b) for a, b in zip((208, 68, 68), (25, 138, 98)))
            box = (x0 + j * cw, y0 + i * ch, x0 + (j + 1) * cw - 25, y0 + (i + 1) * ch - 25)
            draw.rounded_rectangle(box, radius=18, fill=color)
            draw.text((box[0] + 160, box[1] + 48), "NA" if not math.isfinite(value) else f"{value:+.3f}", font=font(30, True), fill=WHITE)

    draw.text((85, 880), "success mean − failure mean（episode等权）", font=font(30, True), fill=INK)
    fw_out = outcome_table[(outcome_table["policy"] == "fastwam") & (outcome_table["L"].isin([3, 5]))]
    table_y = 950
    headers = ["task", "metric", "L=3", "L=5", "解释"]
    widths = [520, 320, 360, 360, 1000]
    x_positions = [80]
    for w in widths[:-1]:
        x_positions.append(x_positions[-1] + w)
    for x, header in zip(x_positions, headers):
        draw.text((x + 10, table_y), header, font=font(23, True), fill=INK)
    row_y = table_y + 55
    for task in tasks:
        for metric in metrics:
            rows = fw_out[(fw_out["task"] == task) & (fw_out["metric"] == metric)]
            vals = {int(row.L): float(row.success_minus_failure) for row in rows.itertuples(index=False)}
            cells = [task, metric, "NA" if not math.isfinite(vals.get(3, float('nan'))) else f"{vals[3]:+.3f}", "NA" if not math.isfinite(vals.get(5, float('nan'))) else f"{vals[5]:+.3f}", "仅描述；adjust没有failure" if task == "adjust_bottle" else "同一批episode，不是两份独立证据"]
            fill = LIGHT if ((row_y - table_y) // 48) % 2 == 0 else WHITE
            draw.rectangle((80, row_y - 5, 2720, row_y + 40), fill=fill)
            for x, cell in zip(x_positions, cells):
                draw.text((x + 10, row_y), str(cell), font=font(19), fill=INK)
            row_y += 48
    path = FIGURES / "reference_04_tail_length_sensitivity.png"
    image.save(path, optimize=True)
    return str(path.relative_to(OUTPUT)).replace("\\", "/")


def metric_action_rows(horizon: pd.DataFrame) -> pd.DataFrame:
    selected = horizon[truthy(horizon["baseline_eligible"]) & truthy(horizon["executed_flag"])].copy()
    selected = selected[selected.apply(lambda row: int(row["L"]) == main_l(str(row["policy"])), axis=1)]
    selected["action_slot"] = pd.to_numeric(selected["action_slot"], errors="coerce")
    selected = selected.dropna(subset=["action_slot"])
    selected["abs_R"] = pd.to_numeric(selected["R_std"], errors="coerce").abs()
    selected["abs_I"] = pd.to_numeric(selected["I_std"], errors="coerce").abs()
    selected["raw_y"] = pd.to_numeric(selected["y_ln"], errors="coerce")
    return selected


def aggregate_time_tables(horizon: pd.DataFrame) -> pd.DataFrame:
    action = metric_action_rows(horizon)
    metrics = {"raw_y": "raw_y", "abs_R": "abs_R", "S": "S_std", "abs_I": "abs_I"}
    rows: list[dict[str, object]] = []
    for (policy, task, success), group in action.groupby(["policy", "task", "success"], sort=True):
        outcome = "success" if bool(success) else "failure"
        for metric_name, column in metrics.items():
            episode_series: dict[str, pd.Series] = {}
            for episode_key, episode_group in group.groupby("episode_key"):
                series = episode_group.groupby("action_slot")[column].mean().sort_index()
                episode_series[str(episode_key)] = series

            # Normalized action progress, 20 bins; first reduce within episode/bin.
            bin_values: dict[int, list[float]] = {idx: [] for idx in range(20)}
            for series in episode_series.values():
                total = float(series.index.max() + 1)
                positions = (series.index.to_numpy(float) + 0.5) / total
                bins = np.minimum((positions * 20).astype(int), 19)
                frame = pd.DataFrame({"bin": bins, "value": series.to_numpy(float)})
                for idx, value in frame.groupby("bin")["value"].mean().items():
                    bin_values[int(idx)].append(float(value))
            for idx, values in bin_values.items():
                arr = np.asarray(values, dtype=float)
                rows.append({
                    "policy": policy, "task": task, "outcome": outcome, "metric": metric_name,
                    "axis": "normalized_progress", "bin_index": idx,
                    "x_left": idx / 20, "x_right": (idx + 1) / 20, "x_center": (idx + 0.5) / 20,
                    "n_at_risk": len(arr), "mean": float(arr.mean()) if len(arr) else float("nan"),
                    "sd": float(arr.std(ddof=1)) if len(arr) > 1 else float("nan"),
                    "episodes_total": len(episode_series), "main_L": main_l(str(policy)),
                })

            max_slot = int(max(series.index.max() for series in episode_series.values()))
            for left in range(0, max_slot + 1, 24 if policy == "fastwam" else 50):
                right = left + (24 if policy == "fastwam" else 50)
                values = [float(series[(series.index >= left) & (series.index < right)].mean()) for series in episode_series.values() if ((series.index >= left) & (series.index < right)).any()]
                arr = np.asarray(values, dtype=float)
                rows.append({
                    "policy": policy, "task": task, "outcome": outcome, "metric": metric_name,
                    "axis": "absolute_slot", "bin_index": left // (24 if policy == "fastwam" else 50),
                    "x_left": left, "x_right": right, "x_center": (left + right) / 2,
                    "n_at_risk": len(arr), "mean": float(arr.mean()) if len(arr) else float("nan"),
                    "sd": float(arr.std(ddof=1)) if len(arr) > 1 else float("nan"),
                    "episodes_total": len(episode_series), "main_L": main_l(str(policy)),
                })
    return pd.DataFrame(rows)


def render_aggregate_figure(policy: str, task: str, metric: str, table: pd.DataFrame) -> str:
    data = table[(table["policy"] == policy) & (table["task"] == task) & (table["metric"] == metric)].copy()
    width, height = 2500, 1500
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    metric_titles = {
        "raw_y": "裸 y（log-DVAC）", "abs_R": "|R|（去位置且标准化后的幅度）",
        "S": "S（query-wide signed）", "abs_I": "|I|（query内局部幅度）",
    }
    draw.text((80, 55), f"总体时间曲线｜{policy} · {TASK_INFO[task]['title']}｜{metric_titles[metric]}", font=font(44, True), fill=INK)
    draw.text((80, 118), f"每条episode先在bin内求均值，再跨episode画 mean ± 1 SD；主口径 L={main_l(policy)}。阴影不是置信区间。", font=font(24), fill=MUTED)

    def panel(axis: str, top: int, bottom: int, title: str) -> None:
        nonlocal draw
        subset = data[data["axis"] == axis]
        ymin, ymax = safe_minmax([subset["mean"] - subset["sd"].fillna(0), subset["mean"] + subset["sd"].fillna(0)], include_zero=metric != "raw_y")
        left, right = 190, 2390
        draw.text((90, top - 55), title, font=font(27, True), fill=INK)
        if subset.empty:
            return
        xmin, xmax = float(subset["x_center"].min()), float(subset["x_center"].max())
        for tick in np.linspace(ymin, ymax, 5):
            py = map_value(float(tick), ymin, ymax, bottom, top)
            draw.line((left, py, right, py), fill=GRID, width=2)
            draw.text((95, py - 11), f"{tick:+.2f}", font=font(18), fill=MUTED)
        if metric != "raw_y" and ymin <= 0 <= ymax:
            py0 = map_value(0, ymin, ymax, bottom, top)
            draw.line((left, py0, right, py0), fill="#68737d", width=3)
        for outcome, color in (("success", GREEN), ("failure", RED)):
            group = subset[subset["outcome"] == outcome].sort_values("x_center")
            if group.empty:
                continue
            points = []
            upper = []
            lower = []
            for row in group.itertuples(index=False):
                px = map_value(float(row.x_center), xmin, xmax, left, right)
                sd = 0.0 if not math.isfinite(float(row.sd)) else float(row.sd)
                points.append((px, map_value(float(row.mean), ymin, ymax, bottom, top)))
                upper.append((px, map_value(float(row.mean + sd), ymin, ymax, bottom, top)))
                lower.append((px, map_value(float(row.mean - sd), ymin, ymax, bottom, top)))
            polygon = upper + list(reversed(lower))
            if polygon:
                overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
                odraw = ImageDraw.Draw(overlay)
                rgba = (25, 138, 98, 48) if outcome == "success" else (208, 68, 68, 48)
                odraw.polygon(polygon, fill=rgba)
                image.paste(overlay, (0, 0), overlay)
                draw = ImageDraw.Draw(image)
            if len(points) > 1:
                draw.line(points, fill=color, width=6, joint="curve")
            for point in points:
                draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=color)
            draw.line((190 + (0 if outcome == "success" else 330), top + 15, 260 + (0 if outcome == "success" else 330), top + 15), fill=color, width=7)
            draw.text((275 + (0 if outcome == "success" else 330), top - 3), "成功" if outcome == "success" else "失败", font=font(20, True), fill=color)
        for tick in np.linspace(xmin, xmax, 9):
            px = map_value(float(tick), xmin, xmax, left, right)
            label = f"{tick:.2f}" if axis == "normalized_progress" else f"{int(round(tick))}"
            draw.text((px - 20, bottom + 14), label, font=font(17), fill=MUTED)

        # n-at-risk directly under each panel.
        risk_y = bottom + 55
        for outcome, color, offset in (("success", GREEN, 0), ("failure", RED, 28)):
            group = subset[subset["outcome"] == outcome].sort_values("x_center")
            if group.empty:
                continue
            draw.text((95, risk_y + offset), f"n {outcome[0].upper()}", font=font(16, True), fill=color)
            for row in group.itertuples(index=False):
                px = map_value(float(row.x_center), xmin, xmax, left, right)
                draw.text((px - 8, risk_y + offset), str(int(row.n_at_risk)), font=font(14), fill=color)

    panel("normalized_progress", 270, 700, "上：各自归一化进度；适合比较相对阶段，但成功终止与400-step timeout不是同一物理时长")
    panel("absolute_slot", 910, 1325, "下：绝对 action slot；n-at-risk下降后，后段主要由长失败episode组成")
    draw.text((80, 1450), TASK_INFO[task]["warning"], font=font(22, True), fill=MUTED)
    path = FIGURES / f"aggregate_{policy}_{task}__{metric}.png"
    image.save(path, optimize=True)
    return str(path.relative_to(OUTPUT)).replace("\\", "/")


def cancellation_figure(horizon: pd.DataFrame) -> str:
    case = next(item for item in CASES if item.slug == "fastwam_move_failure")
    all_l = episode_suffix(horizon, case.episode_uid)
    all_l = all_l[(all_l["policy"] == case.policy) & (all_l["task"] == case.task)].copy()
    all_l = all_l[truthy(all_l["executed_flag"]) & all_l["action_slot"].isin([239, 319])]
    width, height = 2600, 1420
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    draw.text((80, 55), "同一个失败episode的两点：为什么 S 与 I 必须分开看", font=font(46, True), fill=INK)
    draw.text((80, 120), "R = S + I。S 是整条query共同升降；I 是当前future action相对该query的局部偏离。", font=font(26), fill=MUTED)
    positions = [(190, 239, "A", "同向叠加"), (1400, 319, "B", "局部抵消")]
    rows_out: list[dict[str, object]] = []
    for x, slot, letter, title in positions:
        frame = fit_image(read_video_frame(VIDEOS / case.video_name, slot), (960, 540))
        image.paste(frame, (x, 230))
        draw.ellipse((x + 18, 248, x + 76, 306), fill=RED)
        draw.text((x + 36, 257), letter, font=font(29, True), fill=WHITE)
        draw.text((x, 790), f"slot/frame {slot}｜{title}", font=font(30, True), fill=INK)
        bars_y = 900
        for j, l_value in enumerate((3, 5)):
            row = all_l[(all_l["L"] == l_value) & (all_l["action_slot"] == slot)].iloc[0]
            s, i, r = float(row["S_std"]), float(row["I_std"]), float(row["R_std"])
            draw.text((x, bars_y + j * 180), f"L={l_value}", font=font(24, True), fill=INK)
            center = x + 500
            scale = 90
            draw.line((center - 420, bars_y + 45 + j * 180, center + 420, bars_y + 45 + j * 180), fill=GRID, width=3)
            for value, color, label, yy in ((s, GREEN, "S", 0), (i, ORANGE, "I", 42), (r, BLUE, "R=S+I", 84)):
                end = center + value * scale
                draw.line((center, bars_y + 15 + j * 180 + yy, end, bars_y + 15 + j * 180 + yy), fill=color, width=18)
                draw.text((x + 35, bars_y + j * 180 + yy), f"{label} {value:+.3f}", font=font(20, True), fill=color)
            rows_out.append({"slot": slot, "L": l_value, "S_std": s, "I_std": i, "R_std": r})
    draw.text((80, 1335), "用户引用的 +2.645/+1.942/+4.587 与 +1.620/−1.557/+0.063 是旧 L=3；主文同时给论文口径 L=5，避免混口径。", font=font(23, True), fill=MUTED)
    path = FIGURES / "reference_05_move_slot239_319_S_I_cancellation.png"
    image.save(path, optimize=True)
    pd.DataFrame(rows_out).to_csv(OUTPUT / "slot239_319_L3_L5_values.csv", index=False, encoding="utf-8-sig")
    return str(path.relative_to(OUTPUT)).replace("\\", "/")


def aliasing_figure(horizon: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    case_uids = {"episode0004_reset3", "episode0006_reset5"}
    data = horizon[(horizon["policy"] == "fastwam") & (horizon["task"] == "move_stapler_pad") & (horizon["L"] == 5)]
    data = data[data["episode_key"].astype(str).apply(lambda value: any(value.endswith(uid) for uid in case_uids))]
    data = data[truthy(data["executed_flag"])].copy()
    occupancy = data.groupby(["phase_task", "h"]).size().rename("count").reset_index()
    phases = [phase for phase in ["approach", "grasp-contact", "transport-or-rotate", "place-release", "verify", "UNLABELED"] if phase in occupancy["phase_task"].astype(str).unique()]
    matrix = occupancy.pivot_table(index="phase_task", columns="h", values="count", fill_value=0).reindex(phases).reindex(columns=range(24), fill_value=0)
    width, height = 2700, 1600
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    draw.text((80, 55), "位置基线会不会把任务阶段一起过滤掉？答案取决于阶段是否固定落在同一个 h", font=font(44, True), fill=INK)
    draw.text((80, 120), "b_h 只看 chunk 内 future 位置 h；它不知道 episode 时间，也不知道 phase 名称。", font=font(26), fill=MUTED)

    cards = [
        (110, "场景 A：阶段在 query 边界改变", "每个query仍包含 h=0..23。阶段让整条chunk一起升高时，它不会被某个单独 b_h 吃掉，主要保留在 S(q)。", GREEN),
        (1390, "场景 B：阶段总在固定 h 改变", "若跨episode都在同一 h（例如总在h=18接触），b_18会学到这类平均阶段形状，部分模式会被当作位置参照扣除。", ORANGE),
    ]
    for x, title, body, color in cards:
        draw.rounded_rectangle((x, 220, x + 1200, 610), radius=24, fill=LIGHT, outline=color, width=4)
        draw.text((x + 30, 255), title, font=font(31, True), fill=color)
        draw_text_block(draw, (x + 30, 325), body, width_chars=76, size=25, fill=INK, spacing=9)
        for q in range(4):
            qx = x + 40 + q * 275
            if x < 1000:
                draw.rounded_rectangle((qx, 500, qx + 235, 570), radius=10, fill="#dcfce7" if q < 2 else "#fed7aa", outline=WHITE)
                draw.text((qx + 18, 517), f"q{q}: h0…23", font=font(20, True), fill=INK)
            else:
                split = qx + int(235 * 18 / 24)
                draw.rounded_rectangle((qx, 500, qx + 235, 570), radius=10, fill="#dcfce7", outline=WHITE)
                draw.rectangle((split, 500, qx + 235, 570), fill="#fed7aa", outline=WHITE)
                draw.text((qx + 12, 517), f"q{q}", font=font(19, True), fill=INK)
                draw.text((split - 12, 475), "h18", font=font(16, True), fill=ORANGE)

    draw.text((90, 700), "现场两条 move-stapler phase case 的 phase × executed h 占用", font=font(31, True), fill=INK)
    draw.text((90, 748), "若阶段锁死在某个 h，会出现竖向集中；当前大多数阶段在 h0..23 近似平铺，phase 边界主要与 query 边界对齐。", font=font(23), fill=MUTED)
    x0, y0, cell_w, cell_h = 610, 835, 80, 95
    max_count = max(float(matrix.to_numpy().max()), 1.0)
    for i, phase in enumerate(phases):
        draw.text((90, y0 + i * cell_h + 28), phase, font=font(21, True), fill=INK)
        for h in range(24):
            count = float(matrix.loc[phase, h])
            t = count / max_count
            color = tuple(int((1 - t) * a + t * b) for a, b in zip((239, 246, 255), (37, 99, 235)))
            x = x0 + h * cell_w
            y = y0 + i * cell_h
            draw.rectangle((x, y, x + cell_w - 3, y + cell_h - 3), fill=color, outline=WHITE)
            draw.text((x + 28, y + 30), str(int(count)), font=font(17, True), fill=WHITE if t > 0.55 else INK)
    for h in range(24):
        draw.text((x0 + h * cell_w + 25, y0 - 35), str(h), font=font(16), fill=MUTED)
    draw.text((x0 + 830, y0 + len(phases) * cell_h + 25), "future position h", font=font(21, True), fill=MUTED)
    draw.rounded_rectangle((90, 1450, 2610, 1550), radius=18, fill="#fff7ed", outline=ORANGE, width=3)
    draw.text((120, 1480), "因此旧图5只能说明 π0 的固定-h纹理被扣除；它没有逐h阶段标签，不能证明“任务执行模式已被过滤”。", font=font(24, True), fill=INK)
    path = FIGURES / "reference_06_position_phase_aliasing.png"
    image.save(path, optimize=True)
    occupancy.to_csv(OUTPUT / "position_phase_aliasing.csv", index=False, encoding="utf-8-sig")
    return str(path.relative_to(OUTPUT)).replace("\\", "/"), occupancy


def task_overview_figure() -> str:
    width, height = 2500, 1320
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    draw.text((80, 55), "四个 Fast-WAM 任务先看动作目标，再看曲线", font=font(46, True), fill=INK)
    draw.text((80, 120), "帧说明用于理解代表case；除move-stapler两条外，没有独立phase真值。", font=font(25), fill=MUTED)
    tasks = ["adjust_bottle", "move_stapler_pad", "turn_switch", "pick_diverse_bottles"]
    sample_cases = ["fastwam_adjust_success", "fastwam_move_success", "fastwam_turn_success", "fastwam_pick_success"]
    for idx, (task, case_slug) in enumerate(zip(tasks, sample_cases)):
        row, col = divmod(idx, 2)
        x, y = 80 + col * 1210, 220 + row * 520
        case = next(item for item in CASES if item.slug == case_slug)
        frame = fit_image(local_frame(case, case.markers[min(2, len(case.markers)-1)]), (500, 300))
        draw.rounded_rectangle((x, y, x + 1140, y + 460), radius=24, fill=LIGHT, outline=GRID, width=2)
        image.paste(frame, (x + 25, y + 80))
        draw.text((x + 550, y + 28), TASK_INFO[task]["title"], font=font(31, True), fill=INK)
        draw_text_block(draw, (x + 550, y + 85), TASK_INFO[task]["plain"], width_chars=55, size=23, fill=INK, spacing=8)
        draw_text_block(draw, (x + 550, y + 245), TASK_INFO[task]["warning"], width_chars=55, size=20, fill=MUTED, spacing=6)
    path = FIGURES / "reference_00_task_overview.png"
    image.save(path, optimize=True)
    return str(path.relative_to(OUTPUT)).replace("\\", "/")


def write_summary(figure_records: list[dict[str, object]], rank_table: pd.DataFrame, outcome_table: pd.DataFrame, aggregate: pd.DataFrame) -> None:
    summary = {
        "source": str(SOURCE),
        "output": str(OUTPUT),
        "figure_count": len(figure_records),
        "case_count": len(CASES),
        "case_metric_figure_count": sum(record.get("kind") == "case_metric" for record in figure_records),
        "main_L": {"pi0": 3, "fastwam": 5},
        "contracts": {
            "fastwam": "H=32,C<=24,M=10,action_slot==fresh_pre_action_video_frame for executed h<24",
            "pi0": "H=C=50,M=4,query-boundary camera images only",
            "identity_raw": "y=b_h+r",
            "identity_standardized": "R=S_std+I_std",
        },
        "rank_sensitivity_rows": len(rank_table),
        "outcome_sensitivity_rows": len(outcome_table),
        "aggregate_rows": len(aggregate),
    }
    (OUTPUT / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    flat_records = [{key: value for key, value in record.items() if key not in {"frames", "marker_records"}} for record in figure_records]
    pd.DataFrame(flat_records).to_csv(OUTPUT / "figure_index.csv", index=False, encoding="utf-8-sig")
    marker_rows = [row for record in figure_records if record.get("kind") == "case_metric" and record.get("metric") == "R" for row in record.get("marker_records", [])]
    pd.DataFrame(marker_rows).to_csv(OUTPUT / "case_marker_index.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-only", action="store_true", help="Render only the 45 case figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    FRAMES.mkdir(parents=True, exist_ok=True)
    query, horizon, episode, _, _ = load_tables()
    _ = query  # kept in the source contract; horizon drives exact action views.
    figure_records: list[dict[str, object]] = []
    for case in CASES:
        data = prepare_case_data(case, horizon, episode)
        for metric_key in ("raw", "r", "R", "S", "I"):
            figure_records.append(render_case_metric(case, metric_key, data))
    if not args.cases_only:
        for path, kind in (
            (task_overview_figure(), "task_overview"),
            (signal_dictionary_figure(), "signal_dictionary"),
            (denoising_window_figure(), "denoising_window"),
            (alignment_figure(), "alignment"),
        ):
            figure_records.append({"figure": path, "kind": kind})
        figure_records.extend(position_reference_figures(horizon))
        rank_table, outcome_table = sensitivity_tables(horizon)
        rank_table.to_csv(OUTPUT / "tail_length_rank_sensitivity.csv", index=False, encoding="utf-8-sig")
        outcome_table.to_csv(OUTPUT / "tail_length_outcome_sensitivity.csv", index=False, encoding="utf-8-sig")
        figure_records.append({"figure": sensitivity_figure(rank_table, outcome_table), "kind": "tail_sensitivity"})
        figure_records.append({"figure": cancellation_figure(horizon), "kind": "cancellation"})
        alias_path, _ = aliasing_figure(horizon)
        figure_records.append({"figure": alias_path, "kind": "position_phase_aliasing"})
        aggregate = aggregate_time_tables(horizon)
        aggregate.to_csv(OUTPUT / "aggregate_time_curves.csv", index=False, encoding="utf-8-sig")
        for (policy, task), _group in aggregate.groupby(["policy", "task"], sort=True):
            for metric in ("raw_y", "abs_R", "S", "abs_I"):
                figure_records.append({
                    "figure": render_aggregate_figure(str(policy), str(task), metric, aggregate),
                    "kind": "aggregate", "policy": policy, "task": task, "metric": metric,
                })
        write_summary(figure_records, rank_table, outcome_table, aggregate)
    print(json.dumps({"figures": len(figure_records), "output": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
