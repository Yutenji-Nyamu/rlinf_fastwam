from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/current-dvac-grpo-smoke2-20260824"
SUMMARY = EVIDENCE / "runtime/final_summary"
OUT = EVIDENCE / "current_dvac_grpo_smoke2_summary.png"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def line_chart(draw: ImageDraw.ImageDraw, box, series, y_min, y_max, x_labels, title, y_label):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=18, fill="#ffffff", outline="#cbd5e1", width=2)
    draw.text((x0 + 24, y0 + 16), title, font=font(28, True), fill="#0f172a")
    px0, py0, px1, py1 = x0 + 88, y0 + 72, x1 - 28, y1 - 64
    for i in range(5):
        yy = py0 + (py1 - py0) * i / 4
        value = y_max - (y_max - y_min) * i / 4
        draw.line((px0, yy, px1, yy), fill="#e2e8f0", width=1)
        draw.text((x0 + 12, yy - 12), f"{value:.2f}", font=font(17), fill="#475569")
    draw.line((px0, py1, px1, py1), fill="#64748b", width=2)
    draw.text((x0 + 18, y0 + 50), y_label, font=font(16), fill="#64748b")
    n = max(len(values) for _, values, _ in series)
    xs = [px0 + (px1 - px0) * i / max(n - 1, 1) for i in range(n)]
    for i, label in enumerate(x_labels):
        draw.text((xs[i] - 18, py1 + 12), str(label), font=font(18), fill="#475569")
    for label, values, color in series:
        points = []
        for i, value in enumerate(values):
            yy = py1 - (value - y_min) / (y_max - y_min) * (py1 - py0)
            points.append((xs[i], yy))
        if len(points) > 1:
            draw.line(points, fill=color, width=5)
        for point in points:
            draw.ellipse((point[0] - 6, point[1] - 6, point[0] + 6, point[1] + 6), fill=color)
        lx = px0 + 18 + 230 * series.index((label, values, color))
        draw.line((lx, y1 - 28, lx + 34, y1 - 28), fill=color, width=5)
        draw.text((lx + 42, y1 - 42), label, font=font(18), fill="#334155")


steps = json.loads((SUMMARY / "dvac_step_summary.json").read_text(encoding="utf-8"))
with (SUMMARY / "dvac_weight_histogram.csv").open(encoding="utf-8", newline="") as f:
    hist = [row for row in csv.DictReader(f) if int(row["runner_step"]) == 1]
with (SUMMARY / "tensorboard_scalars.csv").open(encoding="utf-8", newline="") as f:
    scalars = list(csv.DictReader(f))
with (EVIDENCE / "runtime/resource.csv").open(encoding="utf-8", newline="") as f:
    resources = [row for row in csv.DictReader(f) if row["driver_alive"] == "1"]


def scalar(tag: str):
    return [float(row["value"]) for row in scalars if row["tag"] == tag]


img = Image.new("RGB", (1600, 1560), "#f1f5f9")
draw = ImageDraw.Draw(img)
draw.text((64, 36), "Current RLinf · π0 · RoboTwin · DVAC-GRPO 两步真实 smoke", font=font(40, True), fill="#0f172a")
draw.text((64, 92), "4×H100；Step 1 建 recent history，Step 2 首次应用 global-z 动作权重", font=font(23), fill="#475569")

# Panel 1: weight distribution.
box = (50, 140, 1550, 570)
draw.rounded_rectangle(box, radius=18, fill="#ffffff", outline="#cbd5e1", width=2)
draw.text((78, 160), "1. Step 2 权重分布（102,400 个 action-slot）", font=font(29, True), fill="#0f172a")
x0, y0, x1, y1 = 115, 240, 1505, 500
max_fraction = max(float(row["fraction"]) for row in hist)
bar_w = (x1 - x0) / len(hist)
for i, row in enumerate(hist):
    frac = float(row["fraction"])
    bx0 = x0 + i * bar_w + 2
    bx1 = x0 + (i + 1) * bar_w - 2
    by0 = y1 - frac / max_fraction * (y1 - y0)
    draw.rectangle((bx0, by0, bx1, y1), fill="#2563eb")
for tick in [0.0, 0.5, 1.0, 1.5, 2.0]:
    xx = x0 + tick / 2.0 * (x1 - x0)
    draw.line((xx, y1, xx, y1 + 8), fill="#475569", width=2)
    draw.text((xx - 18, y1 + 12), f"{tick:.1f}", font=font(18), fill="#475569")
draw.text((1180, 178), "mean 1.002  ·  ESS 0.811", font=font(22, True), fill="#1d4ed8")
draw.text((1180, 207), "P05 0.249  ·  P50 0.966  ·  P95 1.931", font=font(18), fill="#334155")
draw.text((78, 528), "解释：均值仍约为 1，但梯度预算已在 future-action 间重新分配；两端裁剪到 [0, 2]。", font=font(20), fill="#334155")

# Panel 2: two-step training metrics.
line_chart(
    draw,
    (50, 600, 780, 1020),
    [
        ("train success", scalar("env/success_once"), "#059669"),
        ("weight ESS", scalar("train/actor/dvac_weight_ess_fraction"), "#2563eb"),
    ],
    0.0,
    1.0,
    [1, 2],
    "2. 两步训练与加权开关",
    "fraction",
)
line_chart(
    draw,
    (820, 600, 1550, 1020),
    [
        ("approx KL", scalar("train/actor/approx_kl"), "#7c3aed"),
        ("clip fraction", scalar("train/actor/clip_fraction"), "#ea580c"),
    ],
    0.0,
    0.14,
    [1, 2],
    "3. GRPO 优化指标保持有限",
    "value",
)

# Panel 3: resources.
box = (50, 1050, 1550, 1505)
draw.rounded_rectangle(box, radius=18, fill="#ffffff", outline="#cbd5e1", width=2)
draw.text((78, 1072), "4. 资源时间线（1 分钟采样）", font=font(29, True), fill="#0f172a")
px0, py0, px1, py1 = 115, 1148, 1505, 1420
n = len(resources)
xs = [px0 + (px1 - px0) * i / max(n - 1, 1) for i in range(n)]
gpu_mean = [sum(float(row[f"gpu{i}_used_mib"]) for i in range(4)) / 4 / 1024 for row in resources]
host_used = [(2113418768 - float(row["host_mem_available_kib"])) / 1024**2 for row in resources]
for value, label in [(0, "0"), (35, "35"), (70, "70 GiB/GPU")]:
    yy = py1 - value / 70 * (py1 - py0)
    draw.line((px0, yy, px1, yy), fill="#e2e8f0", width=1)
    draw.text((62, yy - 10), label, font=font(17), fill="#475569")
gpu_points = [(xs[i], py1 - min(value, 70) / 70 * (py1 - py0)) for i, value in enumerate(gpu_mean)]
draw.line(gpu_points, fill="#dc2626", width=4)
# Host used is rescaled to the same panel and labeled explicitly.
host_points = [(xs[i], py1 - min(value, 800) / 800 * (py1 - py0)) for i, value in enumerate(host_used)]
draw.line(host_points, fill="#0f766e", width=4)
draw.line((112, 1460, 150, 1460), fill="#dc2626", width=5)
draw.text((160, 1445), "4卡平均显存（左轴，peak 单卡 66.5 GiB）", font=font(18), fill="#334155")
draw.line((690, 1460, 728, 1460), fill="#0f766e", width=5)
draw.text((738, 1445), "主机已用量（按 0–800 GiB 映射；峰约 708 GiB）", font=font(18), fill="#334155")

img.save(OUT, optimize=True)
print(OUT)
