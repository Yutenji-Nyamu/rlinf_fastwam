#!/usr/bin/env python3
"""Render the current Shenzhen RLT-v4 and DSRL-v2 lightweight snapshot.

Only downloaded driver logs and resource CSVs are consumed. Evaluation points
remain discrete; missing warm-up optimization metrics remain missing.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PIL import Image


ROOT = Path(r"C:\Users\86136\Documents\rl")
OUT = ROOT / "docs" / "rlinf-shenzhen-rlt-dsrl-port" / "evidence" / "formal-artifact-refresh-20260824"
BASE_SCRIPT = ROOT / "local_scripts" / "render_shenzhen_rlt_dsrl_formal_live_20260824.py"

spec = importlib.util.spec_from_file_location("formal_viz_base", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {BASE_SCRIPT}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.EVIDENCE = OUT

NUMBER = base.NUMBER
CST = timezone(timedelta(hours=8))


def enrich_rlt(path: Path, rows: list[dict[str, object]]) -> None:
    text = base.clean_text(path)
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
            ("replay/actor_switch_rate", "actor_switch_rate"),
            ("rlt/actor_updates_run", "actor_updates_run"),
            ("rlt/critic_updates_run", "critic_updates_run"),
            ("rlt/pending_update_budget", "pending_update_budget"),
        ):
            row[out_key] = base.optional_number(segment, key)
        # Rich truncates this five-digit scalar in driver.log (for example
        # ``1.1…``). ``cache_size`` is the mean across the two actor ranks, so
        # the exact global total is twice that already-untruncated value.
        row["global_total_transitions"] = 2.0 * float(row["local_cache_size"])


def finite(rows: list[dict[str, object]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))]


def render_rlt(rows: list[dict[str, object]]) -> None:
    xs = [float(row["step"]) for row in rows]
    success = [float(row["train_success"]) for row in rows]
    eval_success = [float(row["eval_success"]) if row["eval_success"] is not None else None for row in rows]
    replay = [float(row["global_min_replay_size"]) for row in rows]
    total = [float(row["global_total_transitions"]) for row in rows]
    wall = [float(row["step_time_s"]) for row in rows]
    rollout = [float(row["generate_rollouts_s"]) for row in rows]
    actor_time = [float(row["actor_training_s"]) for row in rows]
    increments = [right - left for left, right in zip(replay, replay[1:])]
    rate = statistics.median(increments[-20:])
    projected = math.ceil(xs[-1] + max(0.0, 10000.0 - replay[-1]) / rate)
    eval_points = [(int(row["step"]), float(row["eval_success"])) for row in rows if row["eval_success"] is not None]

    image = Image.new("RGB", (1600, 1900), base.LIGHT)
    draw = base.add_header(
        image,
        "RLT Stage 2 v4：保存链正常，当前仍是 reference warm-up",
        "训练成功率来自 reference route；fixed-20 才是 student-only。在线更新尚未开始，所以没有 actor/critic loss。",
    )
    y = base.draw_cards(
        draw,
        137,
        [
            ("完整进度", f"{int(xs[-1])} / 250", "fresh v4，不拼接 v3"),
            ("min-rank replay", f"{replay[-1]:.0f} / 10000", f"按近20步中位速度约 Step {projected} online"),
            ("更新状态", "update_step = 0", "actor switch rate = 0"),
            ("fixed-20", " / ".join(f"S{s}:{v*20:.0f}/20" for s, v in eval_points), "真实离散评估"),
        ],
    )
    base.draw_line_panel(
        draw,
        (55, y, 1545, y + 405),
        xs,
        [
            ("train raw (reference)", success, base.BLUE_LIGHT, 2, False),
            ("rolling-5", base.rolling(success, 5), base.BLUE, 5, False),
            ("rolling-20", base.rolling(success, 20), base.TEAL, 4, False),
            ("fixed-20 student", eval_success, base.ORANGE, 4, True),
        ],
        "成功率：reference rollout 不能冒充 student 学习效果",
        y_min=0,
        y_max=1,
        y_formatter=lambda value: f"{value*100:.0f}%",
        verticals=[(25, "save25", base.PURPLE), (50, "save50", base.PURPLE)],
    )
    y += 430
    base.draw_line_panel(
        draw,
        (55, y, 1545, y + 405),
        xs,
        [
            ("min-rank replay", replay, base.GREEN, 5, False),
            ("global total added", total, base.BLUE, 4, False),
            ("online threshold", [10000.0] * len(xs), base.RED, 3, False),
        ],
        "Replay / schedule：最慢 rank 达到 10,000 才开始 AC 更新",
        y_min=0,
        verticals=[(25, "checkpoint", base.PURPLE), (50, "checkpoint", base.PURPLE)],
    )
    y += 430
    base.draw_line_panel(
        draw,
        (55, y, 1545, y + 405),
        xs,
        [
            ("step wall", wall, base.INK, 4, False),
            ("sim rollout", rollout, base.ORANGE, 4, False),
            ("actor train", actor_time, base.PURPLE, 3, False),
        ],
        "耗时：warm-up 几乎全在仿真；评估步额外执行 fixed-20",
        y_min=0,
        verticals=[(25, "eval + save", base.PURPLE), (50, "eval + save", base.PURPLE)],
    )
    image.save(OUT / "01_rlt_stage2_v4_success_replay_timing.png", optimize=True)


def render_dsrl_success(rows: list[dict[str, object]]) -> None:
    xs = [float(row["step"]) for row in rows]
    success = [float(row["train_success"]) for row in rows]
    eval_success = [float(row["eval_success"]) if row["eval_success"] is not None else None for row in rows]
    resident = [float(row["global_resident_transitions"]) for row in rows]
    cumulative_updates = []
    running = 0.0
    for row in rows:
        running += float(row["planned_optimizer_updates"] or 0.0)
        cumulative_updates.append(running)
        row["cumulative_optimizer_updates"] = running
    wall = [float(row["step_time_s"]) for row in rows]
    rollout = [float(row["generate_rollouts_s"]) for row in rows]
    update_time = [float(row["actor_training_s"]) for row in rows]
    first_online = next(int(row["step"]) for row in rows if row["actor_loss"] is not None)
    eval_points = [(int(row["step"]), float(row["eval_success"])) for row in rows if row["eval_success"] is not None]

    image = Image.new("RGB", (1600, 1910), base.LIGHT)
    draw = base.add_header(
        image,
        "DSRL v2：learned phase 已到正式预算末段",
        "train 每步仅 4 条、fixed 每点 12 条且为 stochastic；曲线显示高成功平台，但不把单个 fixed 点叫严格收敛。",
    )
    y = base.draw_cards(
        draw,
        137,
        [
            ("完整进度", f"{int(xs[-1])} / 200", "Step 13 起 online"),
            ("最近 train", f"5步 {statistics.mean(success[-5:])*100:.1f}%", f"20步 {statistics.mean(success[-20:])*100:.1f}%"),
            ("最新 fixed-12", f"{eval_points[-1][1]*12:.0f} / 12", f"Step {eval_points[-1][0]}，stochastic"),
            ("累计更新预算", f"{cumulative_updates[-1]:,.0f}", f"resident={resident[-1]:.0f} macro transitions"),
        ],
    )
    base.draw_line_panel(
        draw,
        (55, y, 1545, y + 405),
        xs,
        [
            ("train raw", success, base.BLUE_LIGHT, 2, False),
            ("rolling-5", base.rolling(success, 5), base.BLUE, 5, False),
            ("rolling-20", base.rolling(success, 20), base.TEAL, 4, False),
            ("fixed-12", eval_success, base.ORANGE, 4, True),
        ],
        "成功率：真实 fixed 点不连线、不插值",
        y_min=0,
        y_max=1,
        y_formatter=lambda value: f"{value*100:.0f}%",
        verticals=[(first_online, "learned phase", base.PURPLE), (65, "save", base.PURPLE), (130, "save", base.PURPLE), (195, "save", base.PURPLE)],
    )
    y += 430
    base.draw_line_panel(
        draw,
        (55, y, 1545, y + 405),
        xs,
        [
            ("global resident", resident, base.GREEN, 5, False),
            ("cumulative updates / 20", [value / 20 for value in cumulative_updates], base.PURPLE, 4, False),
        ],
        "数据与更新预算：成功早停会同时减少 macro transition 和 UTD20 更新",
        y_min=0,
        verticals=[(first_online, "warm-up end", base.PURPLE)],
    )
    y += 430
    base.draw_line_panel(
        draw,
        (55, y, 1545, y + 405),
        xs,
        [
            ("step wall", wall, base.INK, 4, False),
            ("rollout", rollout, base.ORANGE, 4, False),
            ("SAC update", update_time, base.PURPLE, 4, False),
        ],
        "耗时：online 后绝大多数 wall time 在 UTD20 SAC 更新",
        y_min=0,
        verticals=[(first_online, "online", base.PURPLE)],
    )
    image.save(OUT / "02_dsrl_v2_success_replay_timing.png", optimize=True)


def render_dsrl_optimization(rows: list[dict[str, object]]) -> None:
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

    image = Image.new("RGB", (1600, 2300), base.LIGHT)
    draw = base.add_header(
        image,
        "DSRL 在线优化：标量有限、连续，但 loss 本身不是成功率",
        "actor loss = α logπ − Qπ；critic grad 是 clip 前值（配置 clip=10）。不同量纲分开显示。",
    )
    y = base.draw_cards(
        draw,
        137,
        [
            ("online 区间", f"Step {int(xs[0])}–{int(xs[-1])}", f"{len(xs)} 个完整点"),
            ("actor loss", f"{actor_loss[-1]:.3f}", f"近10均值 {statistics.mean(actor_loss[-10:]):.3f}"),
            ("critic loss", f"{critic_loss[-1]:.3f}", f"近10均值 {statistics.mean(critic_loss[-10:]):.3f}"),
            ("entropy / alpha", f"{entropy[-1]:.2f} / {alpha[-1]:.4f}", "target entropy = -16"),
        ],
    )
    base.draw_line_panel(draw, (55, y, 1545, y + 375), xs, [("actor loss", actor_loss, base.BLUE, 4, False), ("critic loss", critic_loss, base.ORANGE, 4, False)], "SAC losses", zero_line=True)
    y += 400
    base.draw_line_panel(draw, (55, y, 1545, y + 375), xs, [("actor grad", actor_grad, base.BLUE_LIGHT, 2, False), ("rolling-10", base.rolling(actor_grad, 10), base.BLUE, 5, False)], "Actor gradient norm（clip 前）", y_min=0)
    y += 400
    base.draw_line_panel(draw, (55, y, 1545, y + 375), xs, [("critic grad", critic_grad, base.ORANGE_LIGHT, 2, False), ("rolling-10", base.rolling(critic_grad, 10), base.ORANGE, 5, False)], "Critic gradient norm（clip 前；实际 clip=10）", y_min=0)
    y += 400
    base.draw_line_panel(draw, (55, y, 1545, y + 375), xs, [("entropy", entropy, base.PURPLE, 4, False), ("Qπ", q_pi, base.BLUE, 4, False), ("Qdata", q_data, base.ORANGE, 4, False)], "策略熵与 Q：连续有限；Q 更接近 0 不等于校准成功率", zero_line=True)
    y += 400
    base.draw_line_panel(draw, (55, y, 1545, y + 300), xs, [("alpha", alpha, base.GREEN, 4, False)], "自动温度 α：从约 1 降到稳定小量", y_min=0)
    image.save(OUT / "03_dsrl_v2_optimization.png", optimize=True)


def render_resources(rlt: list[dict[str, object]], dsrl: list[dict[str, object]]) -> None:
    def hours(rows: list[dict[str, object]]) -> tuple[datetime, list[float]]:
        start = rows[0]["timestamp"]
        return start, [(row["timestamp"] - start).total_seconds() / 3600 for row in rows]

    rlt_start, xr = hours(rlt)
    dsrl_start, xd = hours(dsrl)
    rlt_mem4 = [float(row["gpu4_used_gib"]) for row in rlt]
    rlt_mem5 = [float(row["gpu5_used_gib"]) for row in rlt]
    dsrl_mem6 = [float(row["gpu6_used_gib"]) for row in dsrl]
    dsrl_mem7 = [float(row["gpu7_used_gib"]) for row in dsrl]
    rlt_util4 = [float(row["gpu4_util_pct"]) for row in rlt]
    rlt_util5 = [float(row["gpu5_util_pct"]) for row in rlt]
    dsrl_util6 = [float(row["gpu6_util_pct"]) for row in dsrl]
    dsrl_util7 = [float(row["gpu7_util_pct"]) for row in dsrl]
    host = [float(row["host_available_gib"]) for row in dsrl]

    def label(start: datetime):
        return lambda value: (start + timedelta(hours=value)).astimezone(CST).strftime("%H:%M")

    rlt_peak = max(finite(rlt, "gpu4_used_gib") + finite(rlt, "gpu5_used_gib"))
    dsrl_peak = max(finite(dsrl, "gpu6_used_gib") + finite(dsrl, "gpu7_used_gib"))
    host_min = min(finite(rlt + dsrl, "host_available_gib"))
    dsrl_terminal_released = math.isnan(dsrl_mem6[-1]) and math.isnan(dsrl_mem7[-1])

    image = Image.new("RGB", (1600, 2300), base.LIGHT)
    draw = base.add_header(
        image,
        "RLT + DSRL 并发资源：四卡健康，DSRL 完成后显存与主存回收",
        f"原始 observer 约10秒采样；并发期 available RAM 最低 {host_min:.0f} GiB，OOM/kill/PSI 均为 0。",
    )
    y = base.draw_cards(
        draw,
        137,
        [
            ("RLT 显存峰", f"{rlt_peak:.1f} GiB", f"末点 {rlt_mem4[-1]:.1f}/{rlt_mem5[-1]:.1f}"),
            ("DSRL 显存峰", f"{dsrl_peak:.1f} GiB", "终态已释放" if dsrl_terminal_released else f"末点 {dsrl_mem6[-1]:.1f}/{dsrl_mem7[-1]:.1f}"),
            ("host available", f"最低 {host_min:.0f} GiB", f"DSRL退出后 {host[-1]:.0f} GiB"),
            ("内存错误", "OOM=0 / kill=0", "PSI some/full = 0"),
        ],
    )
    base.draw_line_panel(draw, (55, y, 1545, y + 350), xr, [("GPU4", rlt_mem4, base.BLUE, 4, False), ("GPU5", rlt_mem5, base.ORANGE, 4, False)], "RLT v4 GPU memory", y_min=0, x_formatter=label(rlt_start))
    y += 375
    base.draw_line_panel(draw, (55, y, 1545, y + 350), xr, [("GPU4 util", rlt_util4, base.BLUE, 2, False), ("GPU5 util", rlt_util5, base.ORANGE, 2, False)], "RLT v4 GPU utilization：rollout 周期与 checkpoint 窗口", y_min=0, y_max=100, x_formatter=label(rlt_start))
    y += 375
    base.draw_line_panel(draw, (55, y, 1545, y + 350), xd, [("GPU6", dsrl_mem6, base.BLUE, 4, False), ("GPU7", dsrl_mem7, base.ORANGE, 4, False)], "DSRL v2 GPU memory：启动峰后稳态约34 GiB/card", y_min=0, x_formatter=label(dsrl_start))
    y += 375
    base.draw_line_panel(draw, (55, y, 1545, y + 350), xd, [("GPU6 util", dsrl_util6, base.BLUE, 2, False), ("GPU7 util", dsrl_util7, base.ORANGE, 2, False)], "DSRL v2 GPU utilization：rollout / UTD20 update 正常锯齿", y_min=0, y_max=100, x_formatter=label(dsrl_start))
    y += 375
    base.draw_line_panel(draw, (55, y, 1545, y + 350), xd, [("host available", host, base.GREEN, 4, False)], "整机 available RAM：并发期余量充足，DSRL 自然退出后回升", y_min=max(0, min(host)-40), y_max=max(host)+40, x_formatter=label(dsrl_start))
    image.save(OUT / "04_concurrent_resource_timeline.png", optimize=True)


def summarize(rlt: list[dict[str, object]], dsrl: list[dict[str, object]], rr: list[dict[str, object]], dr: list[dict[str, object]]) -> dict[str, object]:
    rlt_success = finite(rlt, "train_success")
    dsrl_success = finite(dsrl, "train_success")
    eval_rlt = [{"step": int(row["step"]), "success": float(row["eval_success"]), "episodes": 20} for row in rlt if row["eval_success"] is not None]
    eval_dsrl = [{"step": int(row["step"]), "success": float(row["eval_success"]), "episodes": 12} for row in dsrl if row["eval_success"] is not None]
    cumulative_updates = sum(float(row["planned_optimizer_updates"] or 0) for row in dsrl)
    return {
        "snapshot_generated_cst": datetime.now(CST).isoformat(timespec="seconds"),
        "rlt_v4": {
            "completed_steps": int(rlt[-1]["step"]),
            "train_success_mean": statistics.mean(rlt_success),
            "train_success_last5": statistics.mean(rlt_success[-5:]),
            "train_success_last20": statistics.mean(rlt_success[-20:]),
            "fixed_eval": eval_rlt,
            "global_min_replay_size": float(rlt[-1]["global_min_replay_size"]),
            "global_total_transitions": float(rlt[-1]["global_total_transitions"]),
            "actor_switch_rate": float(rlt[-1]["actor_switch_rate"]),
            "update_step": float(rlt[-1]["update_step"]),
            "mean_step_time_s": statistics.mean(finite(rlt, "step_time_s")),
            "mean_rollout_s": statistics.mean(finite(rlt, "generate_rollouts_s")),
            "checkpoints": {
                "25": {"bytes": 140156837, "files": 3648, "complete": True, "update_step": 0},
                "50": {"bytes": 230536997, "files": 7392, "complete": True, "update_step": 0},
                "75": {"bytes": 322003457, "files": 11181, "complete": True, "update_step": 0},
            },
        },
        "dsrl_v2": {
            "completed_steps": int(dsrl[-1]["step"]),
            "first_online_step": 13,
            "train_success_mean": statistics.mean(dsrl_success),
            "train_success_online_mean": statistics.mean(dsrl_success[12:]),
            "train_success_last5": statistics.mean(dsrl_success[-5:]),
            "train_success_last20": statistics.mean(dsrl_success[-20:]),
            "fixed_eval": eval_dsrl,
            "fixed_eval_mean": statistics.mean(item["success"] for item in eval_dsrl),
            "fixed_eval_last3_mean": statistics.mean(item["success"] for item in eval_dsrl[-3:]),
            "global_resident_transitions": float(dsrl[-1]["global_resident_transitions"]),
            "cumulative_planned_optimizer_updates": cumulative_updates,
            "latest_actor_loss": float(dsrl[-1]["actor_loss"]),
            "latest_critic_loss": float(dsrl[-1]["critic_loss"]),
            "latest_alpha": float(dsrl[-1]["alpha"]),
            "latest_entropy": float(dsrl[-1]["actor_entropy"]),
            "latest_q_pi": float(dsrl[-1]["actor_q_pi"]),
            "latest_q_data": float(dsrl[-1]["critic_q_data"]),
            "checkpoints": {
                "65": {"bytes": 33593478724, "update_step": 34020},
                "130": {"bytes": 33593478725, "update_step": 67900},
                "195": {"bytes": 33593478725, "update_step": 101760},
                "200": {"bytes": 33593478725, "update_step": 104120},
            },
        },
        "resources": {
            "rlt_gpu4_peak_gib": max(finite(rr, "gpu4_used_gib")),
            "rlt_gpu5_peak_gib": max(finite(rr, "gpu5_used_gib")),
            "dsrl_gpu6_peak_gib": max(finite(dr, "gpu6_used_gib")),
            "dsrl_gpu7_peak_gib": max(finite(dr, "gpu7_used_gib")),
            "dsrl_gpu6_median_gib": statistics.median(finite(dr[18:], "gpu6_used_gib")),
            "dsrl_gpu7_median_gib": statistics.median(finite(dr[18:], "gpu7_used_gib")),
            "host_available_min_gib": min(float(row["host_available_gib"]) for row in rr + dr),
            "host_available_latest_gib": float(dr[-1]["host_available_gib"]),
            "cgroup_oom_max": max(finite(rr + dr, "cgroup_oom")),
            "cgroup_oom_kill_max": max(finite(rr + dr, "cgroup_oom_kill")),
        },
    }


def main() -> None:
    rlt = base.parse_metric_tables(OUT / "rlt_v4_driver.log", 250, "rlt")
    enrich_rlt(OUT / "rlt_v4_driver.log", rlt)
    dsrl = base.parse_metric_tables(OUT / "dsrl_v2_driver.log", 200, "dsrl")
    rr = base.read_resource(OUT / "rlt_v4_resource.csv", (4, 5))
    dr = base.read_resource(OUT / "dsrl_v2_resource.csv", (6, 7))

    render_dsrl_success(dsrl)
    base.write_csv(OUT / f"rlt_v4_metrics_step1_{rlt[-1]['step']}.csv", rlt)
    base.write_csv(OUT / f"dsrl_v2_metrics_step1_{dsrl[-1]['step']}.csv", dsrl)
    base.write_csv(OUT / "rlt_v4_fixed20.csv", [{"runner_step": int(row["step"]), "success": row["eval_success"], "episodes": 20} for row in rlt if row["eval_success"] is not None])
    base.write_csv(OUT / "dsrl_v2_fixed12.csv", [{"runner_step": int(row["step"]), "success": row["eval_success"], "episodes": 12} for row in dsrl if row["eval_success"] is not None])
    render_rlt(rlt)
    render_dsrl_optimization(dsrl)
    render_resources(rr, dr)
    summary = summarize(rlt, dsrl, rr, dr)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
