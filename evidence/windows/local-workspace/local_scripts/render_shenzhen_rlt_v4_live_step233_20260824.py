from __future__ import annotations

import importlib.util
import math
import statistics
from pathlib import Path

from PIL import Image


ROOT = Path(r"C:\Users\86136\Documents\rl")
EVIDENCE = ROOT / "docs" / "rlinf-shenzhen-rlt-dsrl-port" / "evidence" / "rlt-v4-live-step232-20260824"
BASE_PATH = ROOT / "local_scripts" / "render_shenzhen_rlt_dsrl_formal_live_20260824.py"
OUT = EVIDENCE / "rlt_stage2_v4_live_step233_overview.png"


def load_base():
    spec = importlib.util.spec_from_file_location("rlt_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    base = load_base()
    rows = base.parse_metric_tables(EVIDENCE / "rlt_v4_driver.log", 250, "rlt")
    resource = base.read_resource(EVIDENCE / "rlt_v4_resource.csv", (4, 5))

    xs = [float(row["step"]) for row in rows]
    train = [float(row["train_success"]) for row in rows]
    eval_values = [float(row["eval_success"]) if row["eval_success"] is not None else None for row in rows]
    step_time = [float(row["step_time_s"]) for row in rows]
    rollout_time = [float(row["generate_rollouts_s"]) if row["generate_rollouts_s"] is not None else None for row in rows]
    actor_time = [float(row["actor_training_s"]) if row["actor_training_s"] is not None else None for row in rows]

    start = resource[0]["timestamp"]
    sampled = [row for index, row in enumerate(resource) if index % 6 == 0]
    resource_x = [(row["timestamp"] - start).total_seconds() / 3600 for row in sampled]
    gpu4_mem = [float(row["gpu4_used_gib"]) for row in sampled]
    gpu5_mem = [float(row["gpu5_used_gib"]) for row in sampled]
    gpu4_util = [float(row["gpu4_util_pct"]) for row in sampled]
    gpu5_util = [float(row["gpu5_util_pct"]) for row in sampled]

    all_gpu4_mem = [float(row["gpu4_used_gib"]) for row in resource if math.isfinite(float(row["gpu4_used_gib"]))]
    all_gpu5_mem = [float(row["gpu5_used_gib"]) for row in resource if math.isfinite(float(row["gpu5_used_gib"]))]
    all_util = [
        (float(row["gpu4_util_pct"]) + float(row["gpu5_util_pct"])) / 2
        for row in resource
        if math.isfinite(float(row["gpu4_util_pct"])) and math.isfinite(float(row["gpu5_util_pct"]))
    ]
    host_available = [float(row["host_available_gib"]) for row in resource]
    latest_eval = next(float(row["eval_success"]) for row in reversed(rows) if row["eval_success"] is not None)
    latest_update = int(float(rows[-1]["update_step"]))
    recent5 = statistics.mean(train[-5:])
    peak_gpu = max(max(all_gpu4_mem), max(all_gpu5_mem))

    image = Image.new("RGB", (1600, 2000), base.LIGHT)
    draw = base.add_header(
        image,
        "RLT Stage 2 v4：当前完整到 Step 233 / 250",
        "2×H100（物理 GPU 4–5），8 env；Step 225 fixed-20 已达 19/20。曲线只使用现场原始日志。",
    )
    y = base.draw_cards(
        draw,
        137,
        [
            ("当前", "233 / 250", "19:05 现场已继续到 Step 235"),
            ("最近训练成功率", f"{recent5:.1%}", "最近 5 个完整 outer step"),
            ("fixed-20", f"{int(latest_eval * 20)}/20", "Step 225；95%"),
            ("在线更新", f"{latest_update:,}", "update_step；无非有限值"),
            ("单卡显存峰值", f"{peak_gpu:.1f} GiB", "80 GiB H100；无 OOM"),
        ],
    )

    base.draw_line_panel(
        draw,
        (55, y, 1545, y + 370),
        xs,
        [
            ("train / step", train, base.GRAY, 2, False),
            ("train MA(5)", base.rolling(train, 5), base.GREEN, 5, False),
            ("fixed-20 eval", eval_values, base.ORANGE, 3, True),
        ],
        "成功率：warm-up 后进入在线更新，fixed-20 从 0 升至 95%",
        y_min=0,
        y_max=1.05,
        verticals=[(154, "online update begins", base.PURPLE)],
    )
    y += 395
    base.draw_line_panel(
        draw,
        (55, y, 1545, y + 370),
        xs,
        [
            ("total MA(5)", base.rolling(step_time, 5), base.INK, 5, False),
            ("rollout MA(5)", base.rolling(rollout_time, 5), base.BLUE, 4, False),
            ("actor update MA(5)", base.rolling(actor_time, 5), base.RED, 4, False),
        ],
        "每步耗时分解：主要仍是 RoboTwin rollout；在线更新约十几秒",
        y_min=0,
        verticals=[(154, "online update begins", base.PURPLE)],
    )
    y += 395
    base.draw_line_panel(
        draw,
        (55, y, 1545, y + 310),
        resource_x,
        [
            ("GPU 4", gpu4_mem, base.BLUE, 4, False),
            ("GPU 5", gpu5_mem, base.PURPLE, 4, False),
        ],
        "显存：两卡稳定在约 18–22 GiB；容量不是使用两卡的原因",
        y_min=0,
        y_max=80,
        x_formatter=lambda value: f"{value:.1f}h",
    )
    y += 335
    base.draw_line_panel(
        draw,
        (55, y, 1545, y + 310),
        resource_x,
        [
            ("GPU 4 util", gpu4_util, base.BLUE, 3, False),
            ("GPU 5 util", gpu5_util, base.PURPLE, 3, False),
        ],
        f"利用率：10 秒原始采样；两卡均值约 {statistics.mean(all_util):.1f}%（rollout 主导，呈脉冲）",
        y_min=0,
        y_max=100,
        x_formatter=lambda value: f"{value:.1f}h",
    )

    draw.text(
        (58, 1940),
        (
            f"资源结论：主机 MemAvailable 最低 {min(host_available):.0f} GiB；cgroup OOM kill=0。"
            "单卡容量足够，但改为 1 rank 会改变 world-size、梯度累积和 replay 合同；当前可比 baseline 保持 2 卡。"
        ),
        font=base.F_NOTE,
        fill=base.MUTED,
    )
    image.save(OUT, optimize=True)
    print(OUT)


if __name__ == "__main__":
    main()
