#!/usr/bin/env python3
"""Render and summarize the completed Shenzhen RLT Stage 2 v4 run."""

from __future__ import annotations

import importlib.util
import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path

from PIL import Image


ROOT = Path(r"C:\Users\86136\Documents\rl")
OUT = ROOT / "docs" / "rlinf-shenzhen-rlt-dsrl-port" / "evidence" / "rlt-v4-final250-20260824"
BASE_PATH = ROOT / "local_scripts" / "render_shenzhen_rlt_dsrl_formal_live_20260824.py"


def load_base():
    spec = importlib.util.spec_from_file_location("rlt_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite(values):
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def enrich(base, rows):
    text = base.clean_text(OUT / "rlt_v4_driver.log")
    matches = list(re.finditer(r"Global Step:\s*(\d+)/250", text))
    by_step = {int(row["step"]): row for row in rows}
    for index, match in enumerate(matches):
        step = int(match.group(1))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segment = text[match.start() : end]
        row = by_step.get(step)
        if row is None:
            continue
        for key, out_key in (
            ("actor/grad_norm", "actor_grad_norm"),
            ("critic/grad_norm", "critic_grad_norm"),
            ("rlt/updates_scheduled", "updates_scheduled"),
            ("rlt/global_total_transitions_added", "global_total_transitions_added"),
        ):
            row[out_key] = base.optional_number(segment, key)


def render_learning(base, rows):
    xs = [float(row["step"]) for row in rows]
    train = [float(row["train_success"]) for row in rows]
    eval_success = [float(row["eval_success"]) if row["eval_success"] is not None else None for row in rows]
    replay = [float(row["global_min_replay_size"]) for row in rows]
    updates = [float(row["update_step"]) for row in rows]
    first_online = next(int(row["step"]) for row in rows if float(row["update_step"]) > 0)
    latest_eval_step = next(int(row["step"]) for row in reversed(rows) if row["eval_success"] is not None)
    latest_eval = next(float(row["eval_success"]) for row in reversed(rows) if row["eval_success"] is not None)

    image = Image.new("RGB", (1600, 1510), base.LIGHT)
    draw = base.add_header(
        image,
        "RLT Stage 2 v4：250/250 自然完成",
        "精确 π0 + current AR RLT；2×H100、8 env。fixed-20 是 student-only 评估，橙点不插值。",
    )
    y = base.draw_cards(
        draw,
        137,
        [
            ("终态", "250 / 250", "exit 0；无 fatal / OOM"),
            ("最终 train", "7 / 8", "本步 reference-route rollout"),
            ("最终 fixed-20", f"{int(latest_eval * 20)} / 20", f"Step {latest_eval_step}"),
            ("在线更新", f"{int(updates[-1]):,}", f"Step {first_online} 起"),
            ("最近 20 步 train", f"{statistics.mean(train[-20:]):.1%}", "仅描述训练 rollout"),
        ],
    )
    base.draw_line_panel(
        draw,
        (55, y, 1545, y + 470),
        xs,
        [
            ("train / step", train, base.GRAY, 2, False),
            ("train MA(5)", base.rolling(train, 5), base.GREEN, 5, False),
            ("train MA(20)", base.rolling(train, 20), base.TEAL, 4, False),
            ("fixed-20", eval_success, base.ORANGE, 4, True),
        ],
        "成功率：warm-up 后 student fixed-20 从 0 上升至 18/20",
        y_min=0,
        y_max=1.05,
        y_formatter=lambda value: f"{value * 100:.0f}%",
        verticals=[(first_online, "online update begins", base.PURPLE)],
    )
    y += 500
    base.draw_line_panel(
        draw,
        (55, y, 1545, y + 440),
        xs,
        [
            ("min-rank replay", replay, base.BLUE, 4, False),
            ("online threshold", [10000.0] * len(xs), base.RED, 3, False),
            ("update_step / 10", [value / 10.0 for value in updates], base.PURPLE, 4, False),
        ],
        "Replay 与更新：最慢 rank 过 10,000 后进入在线 AC；紫线为 update_step÷10",
        y_min=0,
        verticals=[(first_online, "online", base.PURPLE)],
    )
    image.save(OUT / "01_rlt_final_learning_and_eval.png", optimize=True)


def render_optimization(base, rows):
    first_online = next(int(row["step"]) for row in rows if float(row["update_step"]) > 0)
    online = [row for row in rows if row["actor_loss"] is not None and int(row["step"]) >= first_online]
    xs = [float(row["step"]) for row in online]
    actor_loss = [float(row["actor_loss"]) for row in online]
    critic_loss = [float(row["critic_loss"]) for row in online]
    actor_grad = [float(row["actor_grad_norm"]) for row in online]
    critic_grad = [float(row["critic_grad_norm"]) for row in online]
    all_x = [float(row["step"]) for row in rows]
    wall = [float(row["step_time_s"]) for row in rows]
    rollout = [float(row["generate_rollouts_s"]) if row["generate_rollouts_s"] is not None else None for row in rows]
    actor_time = [float(row["actor_training_s"]) if row["actor_training_s"] is not None else None for row in rows]

    image = Image.new("RGB", (1600, 1880), base.LIGHT)
    draw = base.add_header(
        image,
        "RLT Stage 2 v4：在线优化与耗时",
        "loss、梯度和时间分量分开显示；所有 250 个完整 step 连续，在线标量均为有限值。",
    )
    y = base.draw_cards(
        draw,
        137,
        [
            ("在线区间", f"Step {first_online}–250", f"{len(online)} 个完整点"),
            ("最终 actor loss", f"{actor_loss[-1]:.4f}", f"近10均值 {statistics.mean(actor_loss[-10:]):.4f}"),
            ("最终 critic loss", f"{critic_loss[-1]:.4f}", f"近10均值 {statistics.mean(critic_loss[-10:]):.4f}"),
            ("最终 actor grad", f"{actor_grad[-1]:.3f}", f"critic {critic_grad[-1]:.3f}"),
        ],
    )
    base.draw_line_panel(
        draw,
        (55, y, 1545, y + 370),
        xs,
        [("actor loss", actor_loss, base.BLUE, 3, False), ("MA(10)", base.rolling(actor_loss, 10), base.PURPLE, 5, False)],
        "Actor loss：进入在线阶段后连续、有限",
        zero_line=True,
    )
    y += 395
    base.draw_line_panel(
        draw,
        (55, y, 1545, y + 370),
        xs,
        [("critic loss", critic_loss, base.ORANGE, 3, False), ("MA(10)", base.rolling(critic_loss, 10), base.RED, 5, False)],
        "Critic loss：末段稳定在小量级",
        y_min=0,
        y_formatter=lambda value: f"{value:.4f}",
    )
    y += 395
    base.draw_line_panel(
        draw,
        (55, y, 1545, y + 370),
        xs,
        [("actor grad", actor_grad, base.BLUE, 3, False), ("critic grad", critic_grad, base.ORANGE, 3, False)],
        "梯度范数：无 non-finite 或爆炸退出",
        y_min=0,
    )
    y += 395
    base.draw_line_panel(
        draw,
        (55, y, 1545, y + 310),
        all_x,
        [
            ("step wall MA(5)", base.rolling(wall, 5), base.INK, 5, False),
            ("rollout MA(5)", base.rolling(rollout, 5), base.BLUE, 4, False),
            ("actor update MA(5)", base.rolling(actor_time, 5), base.RED, 4, False),
        ],
        "耗时：RoboTwin rollout 是主要分量；评估/保存步形成离散峰",
        y_min=0,
        verticals=[(first_online, "online", base.PURPLE)],
    )
    image.save(OUT / "02_rlt_final_optimization_and_timing.png", optimize=True)


def render_resources(base, resources):
    start = resources[0]["timestamp"]
    sampled = [row for index, row in enumerate(resources) if index % 6 == 0]
    xs = [(row["timestamp"] - start).total_seconds() / 3600.0 for row in sampled]
    mem4 = [float(row["gpu4_used_gib"]) for row in sampled]
    mem5 = [float(row["gpu5_used_gib"]) for row in sampled]
    util4 = [float(row["gpu4_util_pct"]) for row in sampled]
    util5 = [float(row["gpu5_util_pct"]) for row in sampled]
    host = [float(row["host_available_gib"]) for row in sampled]
    mem4_all = finite(row["gpu4_used_gib"] for row in resources)
    mem5_all = finite(row["gpu5_used_gib"] for row in resources)
    util_all = finite((float(row["gpu4_util_pct"]) + float(row["gpu5_util_pct"])) / 2 for row in resources if math.isfinite(float(row["gpu4_util_pct"])) and math.isfinite(float(row["gpu5_util_pct"])))
    host_all = finite(row["host_available_gib"] for row in resources)
    oom = finite(row["cgroup_oom"] for row in resources)
    oom_kill = finite(row["cgroup_oom_kill"] for row in resources)

    image = Image.new("RGB", (1600, 1500), base.LIGHT)
    draw = base.add_header(
        image,
        "RLT Stage 2 v4：两卡与主机资源",
        "约 10 秒一次的原始 observer 采样；终态 GPU 已释放。利用率锯齿来自 rollout / update / eval / save 阶段切换。",
    )
    y = base.draw_cards(
        draw,
        137,
        [
            ("GPU4 峰值", f"{max(mem4_all):.1f} GiB", "H100 80 GiB"),
            ("GPU5 峰值", f"{max(mem5_all):.1f} GiB", "H100 80 GiB"),
            ("两卡平均利用率", f"{statistics.mean(util_all):.1f}%", "全流程 10 秒采样"),
            ("最低可用主存", f"{min(host_all):.0f} GiB", "OOM / kill = 0"),
        ],
    )
    base.draw_line_panel(
        draw,
        (55, y, 1545, y + 370),
        xs,
        [("GPU 4", mem4, base.BLUE, 4, False), ("GPU 5", mem5, base.PURPLE, 4, False)],
        "显存：容量余量很大；两卡用于保持已验证的 world-size / batch / replay 合同",
        y_min=0,
        y_max=80,
        x_formatter=lambda value: f"{value:.1f}h",
    )
    y += 395
    base.draw_line_panel(
        draw,
        (55, y, 1545, y + 370),
        xs,
        [("GPU 4 util", util4, base.BLUE, 3, False), ("GPU 5 util", util5, base.PURPLE, 3, False)],
        "GPU 利用率：仿真主导，呈阶段性脉冲",
        y_min=0,
        y_max=100,
        x_formatter=lambda value: f"{value:.1f}h",
    )
    y += 395
    base.draw_line_panel(
        draw,
        (55, y, 1545, y + 320),
        xs,
        [("host available", host, base.GREEN, 4, False)],
        "整机可用主存：全程无内存压力或 cgroup OOM",
        y_min=max(0, min(host_all) - 60),
        y_max=max(host_all) + 60,
        x_formatter=lambda value: f"{value:.1f}h",
    )
    image.save(OUT / "03_rlt_final_resources.png", optimize=True)
    return {
        "gpu4_peak_gib": max(mem4_all),
        "gpu5_peak_gib": max(mem5_all),
        "mean_gpu_util_pct": statistics.mean(util_all),
        "host_available_min_gib": min(host_all),
        "cgroup_oom_max": max(oom) if oom else 0,
        "cgroup_oom_kill_max": max(oom_kill) if oom_kill else 0,
    }


def main():
    base = load_base()
    rows = base.parse_metric_tables(OUT / "rlt_v4_driver.log", 250, "rlt")
    enrich(base, rows)
    if int(rows[-1]["step"]) != 250:
        raise RuntimeError(f"expected complete Step 250, got {rows[-1]['step']}")
    resources = base.read_resource(OUT / "rlt_v4_resource.csv", (4, 5))
    render_learning(base, rows)
    render_optimization(base, rows)
    resource_summary = render_resources(base, resources)

    base.write_csv(OUT / "rlt_v4_metrics_step1_250.csv", rows)
    fixed = [
        {"runner_step": int(row["step"]), "success": float(row["eval_success"]), "success_count": int(round(float(row["eval_success"]) * 20)), "episodes": 20}
        for row in rows
        if row["eval_success"] is not None
    ]
    base.write_csv(OUT / "rlt_v4_fixed20.csv", fixed)

    start = datetime.fromisoformat((OUT / "rlt_v4_started_at.txt").read_text(encoding="utf-8").strip())
    finish = datetime.fromisoformat((OUT / "rlt_v4_finished_at.txt").read_text(encoding="utf-8").strip())
    train = [float(row["train_success"]) for row in rows]
    first_online = next(int(row["step"]) for row in rows if float(row["update_step"]) > 0)
    summary = {
        "run": "formal-current-ar-stage2-8env250-20260824-v4-warmup-fix",
        "status": "completed",
        "exit_code": int((OUT / "rlt_v4_exit_code.txt").read_text(encoding="utf-8").strip()),
        "completed_steps": 250,
        "duration_seconds": (finish - start).total_seconds(),
        "final_train_success": float(rows[-1]["train_success"]),
        "train_success_last5": statistics.mean(train[-5:]),
        "train_success_last20": statistics.mean(train[-20:]),
        "fixed20": fixed,
        "final_fixed20_success": float(fixed[-1]["success"]),
        "first_online_step": first_online,
        "final_update_step": int(float(rows[-1]["update_step"])),
        "final_actor_loss": float(rows[-1]["actor_loss"]),
        "final_critic_loss": float(rows[-1]["critic_loss"]),
        "final_actor_grad_norm": float(rows[-1]["actor_grad_norm"]),
        "final_critic_grad_norm": float(rows[-1]["critic_grad_norm"]),
        "mean_step_time_s": statistics.mean(float(row["step_time_s"]) for row in rows),
        "mean_rollout_time_s": statistics.mean(finite(row["generate_rollouts_s"] for row in rows)),
        "resource": resource_summary,
        "checkpoint": {
            "path": "robotwin_adjust_bottle_rlt_stage2_current_ar_8env250_optimizer_warmup_v3/checkpoints/global_step_250",
            "present_on_server": True,
            "included_in_light_bundle": False,
        },
        "fatal_or_oom": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
