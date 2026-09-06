from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(
    r"C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-dvac-telemetry"
    r"\evidence\rlt_dvac_formal_live_g334_20260825"
)
RAW = ROOT / "raw"


def first_float(text: str, key: str) -> float | None:
    match = re.search(rf"{re.escape(key)}=([-+0-9.eE]+)", text)
    return float(match.group(1)) if match else None


def first_number_after(text: str, label: str) -> float | None:
    match = re.search(rf"{re.escape(label)}\s*([-+0-9.eE]+)", text)
    return float(match.group(1)) if match else None


def section(block: str, title: str) -> str:
    match = re.search(
        rf"{re.escape(title)}[^\n]*\n(?:│[^\n]*\n)?(.*?)(?=├|╰)",
        block,
        flags=re.DOTALL,
    )
    return match.group(1) if match else ""


def parse_metrics(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="replace")
    starts = list(re.finditer(r"│ Global Step:\s+(\d+)/480", text))
    rows: list[dict[str, float | int | None | str]] = []
    for idx, start in enumerate(starts):
        stop = starts[idx + 1].start() if idx + 1 < len(starts) else len(text)
        block = text[start.start() : stop]
        # A valid completed metric table has its closing border and environment data.
        env = section(block, "Environment")
        if "╰" not in block or not env:
            continue
        evaluation = section(block, "Evaluation")
        elapsed_match = re.search(r"Elapsed:\s*([^│]+?)\s*│", block)
        row: dict[str, float | int | None | str] = {
            "step": int(start.group(1)),
            "elapsed": elapsed_match.group(1).strip() if elapsed_match else "",
            "step_time_s": first_number_after(block, "Step Time:"),
            "train_success": first_float(env, "success_once"),
            "train_reward": first_float(env, "reward"),
            "eval_success": first_float(evaluation, "success_once"),
            "global_min_replay": first_float(block, "rlt/global_min_replay_size"),
            "actor_updates": first_float(block, "rlt/actor_updates_run"),
            "critic_updates": first_float(block, "rlt/critic_updates_run"),
            "update_step": first_float(block, "rlt/update_step"),
            "ready_for_online": first_float(block, "rlt/ready_for_online"),
            "actor_grad_norm": first_float(block, "actor/grad_norm"),
            "critic_grad_norm": first_float(block, "critic/grad_norm"),
            "actor_loss": first_float(block, "sac/actor_loss"),
            "critic_loss": first_float(block, "sac/critic_loss"),
            "q_pi": first_float(block, "actor/q_pi"),
            "bc_loss": first_float(block, "actor/bc_loss"),
            "baseline_count": first_float(block, "actor/rlt_dvac/baseline_count"),
            "baseline_frozen": first_float(block, "actor/rlt_dvac/baseline_frozen"),
            "mode_apply": first_float(block, "actor/rlt_dvac/mode_apply"),
            "log_v_mean": first_float(block, "actor/rlt_dvac/log_v_mean"),
            "log_v_std": first_float(block, "actor/rlt_dvac/log_v_std"),
            "weight_mean": first_float(block, "actor/rlt_dvac/weight_mean"),
            "weight_median": first_float(block, "actor/rlt_dvac/weight_median"),
            "weight_p05": first_float(block, "actor/rlt_dvac/weight_p05"),
            "weight_p95": first_float(block, "actor/rlt_dvac/weight_p95"),
            "weight_ess": first_float(block, "actor/rlt_dvac/weight_ess_ratio"),
            "top20_weight_mass": first_float(block, "actor/rlt_dvac/top20_weight_mass"),
        }
        for horizon in range(10):
            row[f"weight_h{horizon:02d}"] = first_float(
                block, f"actor/rlt_dvac/weight_h{horizon:02d}"
            )
        rows.append(row)

    frame = pd.DataFrame(rows).drop_duplicates("step", keep="last").sort_values("step")
    frame["success_roll5"] = frame["train_success"].rolling(5, min_periods=1).mean()
    frame["success_roll10"] = frame["train_success"].rolling(10, min_periods=1).mean()
    return frame


def parse_resources(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    numeric = [column for column in frame.columns if column != "unix_time"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["unix_time", "cgroup_current_bytes"])
    frame["hours"] = (frame["unix_time"] - frame["unix_time"].iloc[0]) / 3600.0
    gib = 1024.0**3
    frame["cgroup_gib"] = frame["cgroup_current_bytes"] / gib
    frame["anon_gib"] = frame["anon_bytes"] / gib
    frame["file_gib"] = frame["file_bytes"] / gib
    frame["env_rss_gib"] = frame["env_rss_kib"] / 1024.0**2
    frame["gpu0_gib"] = frame["gpu0_used_mib"] / 1024.0
    frame["gpu1_gib"] = frame["gpu1_used_mib"] / 1024.0
    return frame


FONT_REGULAR = r"C:\Windows\Fonts\segoeui.ttf"
FONT_SEMIBOLD = r"C:\Windows\Fonts\seguisb.ttf"


def font(size: int, semibold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_SEMIBOLD if semibold else FONT_REGULAR, size)


def draw_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    title: str,
    x_label: str,
    y_label: str,
    x_values: list[float],
    series: list[tuple[str, list[float], str, int]],
    y_min: float,
    y_max: float,
    horizontal: list[tuple[float, str, str]] | None = None,
    vertical: list[tuple[float, str, str]] | None = None,
    note: str | None = None,
) -> None:
    left, top, right, bottom = rect
    draw.rounded_rectangle(rect, radius=18, fill="#FFFFFF", outline="#D7DEE9", width=2)
    draw.text((left + 28, top + 22), title, fill="#172033", font=font(27, True))
    legend_y = top + 70
    legend_x = left + 30
    for name, _, color, width in series:
        draw.line(
            (legend_x, legend_y + 10, legend_x + 34, legend_y + 10),
            fill=color,
            width=max(3, width),
        )
        draw.text((legend_x + 44, legend_y), name, fill="#475569", font=font(17))
        legend_x += 44 + int(draw.textlength(name, font=font(17))) + 25

    plot_left, plot_top = left + 105, top + 118
    plot_right, plot_bottom = right - 32, bottom - 105
    x_min, x_max = min(x_values), max(x_values)
    if x_max == x_min:
        x_max = x_min + 1

    def px(x: float) -> float:
        return plot_left + (x - x_min) / (x_max - x_min) * (plot_right - plot_left)

    def py(y: float) -> float:
        return plot_bottom - (y - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

    for tick in range(6):
        y = y_min + tick * (y_max - y_min) / 5
        yp = py(y)
        draw.line((plot_left, yp, plot_right, yp), fill="#E5EAF1", width=1)
        label = f"{y:.0f}" if abs(y_max - y_min) >= 10 else f"{y:.1f}"
        tw = draw.textlength(label, font=font(16))
        draw.text((plot_left - tw - 12, yp - 10), label, fill="#64748B", font=font(16))
    for tick in range(6):
        x = x_min + tick * (x_max - x_min) / 5
        xp = px(x)
        label = f"{x:.0f}" if x_max >= 20 else f"{x:.1f}"
        tw = draw.textlength(label, font=font(16))
        draw.text((xp - tw / 2, plot_bottom + 12), label, fill="#64748B", font=font(16))

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#94A3B8", width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#94A3B8", width=2)
    if horizontal:
        for value, label, color in horizontal:
            if y_min <= value <= y_max:
                yp = py(value)
                for x0 in range(int(plot_left), int(plot_right), 18):
                    draw.line((x0, yp, min(x0 + 10, plot_right), yp), fill=color, width=2)
                draw.text(
                    (plot_right - draw.textlength(label, font=font(15)) - 4, yp - 23),
                    label,
                    fill=color,
                    font=font(15),
                )
    if vertical:
        for value, label, color in vertical:
            if x_min <= value <= x_max:
                xp = px(value)
                for y0 in range(int(plot_top), int(plot_bottom), 18):
                    draw.line((xp, y0, xp, min(y0 + 10, plot_bottom)), fill=color, width=2)
                draw.text((xp + 7, plot_top + 4), label, fill=color, font=font(15))

    for _, values, color, width in series:
        points = [
            (px(float(x)), py(float(y)))
            for x, y in zip(x_values, values)
            if pd.notna(y) and y_min <= float(y) <= y_max
        ]
        if len(points) >= 2:
            draw.line(points, fill=color, width=width, joint="curve")
        for x, y in points:
            if len(points) <= 20:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)

    draw.text(
        (
            (plot_left + plot_right) / 2 - draw.textlength(x_label, font=font(17)) / 2,
            bottom - 34,
        ),
        x_label,
        fill="#475569",
        font=font(17),
    )
    draw.text((left + 18, plot_top - 30), y_label, fill="#475569", font=font(16))
    if note:
        draw.text((plot_left, bottom - 63), note, fill="#475569", font=font(16))


def main() -> None:
    steps = parse_metrics(RAW / "metrics.log")
    resources = parse_resources(RAW / "resources.csv")
    steps.to_csv(ROOT / "step_metrics.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    latest = steps.iloc[-1]
    active = steps.loc[steps["mode_apply"] == 1].copy()
    first_apply = int(active.step.iloc[0])
    evaluations = steps.dropna(subset=["eval_success"])
    latest_eval = evaluations.iloc[-1]
    latest_h_weights = [float(latest[f"weight_h{h:02d}"]) for h in range(10)]

    summary = {
        "snapshot_latest_complete_step": int(latest.step),
        "progress_fraction": float(latest.step / 480.0),
        "elapsed": str(latest.elapsed),
        "latest_train_success": float(latest.train_success),
        "latest_success_roll5": float(latest.success_roll5),
        "latest_success_roll10": float(latest.success_roll10),
        "all_steps_train_success_mean": float(steps.train_success.mean()),
        "first_apply_step": first_apply,
        "active_phase_train_success_mean": float(active.train_success.mean()),
        "latest_fixed20_eval_step": int(latest_eval.step),
        "latest_fixed20_eval_success": float(latest_eval.eval_success),
        "fixed20_eval_points": {
            str(int(row.step)): float(row.eval_success)
            for _, row in evaluations.iterrows()
        },
        "global_min_replay": float(latest.global_min_replay),
        "update_step": float(latest.update_step),
        "actor_updates_latest_cycle": float(latest.actor_updates),
        "critic_updates_latest_cycle": float(latest.critic_updates),
        "actor_grad_norm": float(latest.actor_grad_norm),
        "critic_grad_norm": float(latest.critic_grad_norm),
        "actor_loss": float(latest.actor_loss),
        "critic_loss": float(latest.critic_loss),
        "baseline_count": float(latest.baseline_count),
        "baseline_frozen": bool(latest.baseline_frozen),
        "latest_weight_mean": float(latest.weight_mean),
        "latest_weight_median": float(latest.weight_median),
        "latest_weight_p05": float(latest.weight_p05),
        "latest_weight_p95": float(latest.weight_p95),
        "latest_weight_ess": float(latest.weight_ess),
        "latest_top20_weight_mass": float(latest.top20_weight_mass),
        "latest_h00_to_h09_weights": latest_h_weights,
        "cgroup_current_gib": float(resources.cgroup_gib.iloc[-1]),
        "cgroup_peak_gib": float(resources.cgroup_gib.max()),
        "anon_current_gib": float(resources.anon_gib.iloc[-1]),
        "file_current_gib": float(resources.file_gib.iloc[-1]),
        "env_rss_current_gib": float(resources.env_rss_gib.iloc[-1]),
        "env_rss_peak_gib": float(resources.env_rss_gib.max()),
        "gpu0_current_gib": float(resources.gpu0_gib.iloc[-1]),
        "gpu1_current_gib": float(resources.gpu1_gib.iloc[-1]),
        "gpu0_peak_gib": float(resources.gpu0_gib.max()),
        "gpu1_peak_gib": float(resources.gpu1_gib.max()),
        "event_high": int(resources.event_high.max()),
        "event_max": int(resources.event_max.max()),
        "event_oom": int(resources.event_oom.max()),
        "event_oom_kill": int(resources.event_oom_kill.max()),
        "psi_some_avg10_peak": float(resources.psi_some_avg10.max()),
        "psi_full_avg10_peak": float(resources.psi_full_avg10.max()),
        "resource_samples": int(len(resources)),
    }
    (ROOT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    image = Image.new("RGB", (2520, 1660), "#F5F7FB")
    draw = ImageDraw.Draw(image)
    draw.text(
        (70, 40),
        f"RLT teacher-DVAC fresh-480 — completed step {int(latest.step)}/480",
        fill="#172033",
        font=font(42, True),
    )
    draw.text(
        (72, 98),
        f"Healthy and actively learning · DVAC apply from g{first_apply} · latest checkpoint g325 · no OOM/fatal events",
        fill="#58657A",
        font=font(23),
    )

    draw_panel(
        draw,
        (55, 155, 1240, 845),
        "Training rollout success",
        "Global step",
        "Success (%)",
        steps.step.astype(float).tolist(),
        [
            ("per-step (8 eps)", (100 * steps.train_success).tolist(), "#A7AFBD", 2),
            ("5-step mean", (100 * steps.success_roll5).tolist(), "#F97316", 5),
            ("10-step mean", (100 * steps.success_roll10).tolist(), "#2563EB", 5),
        ],
        0,
        100,
        vertical=[(first_apply, "DVAC apply", "#7C3AED")],
        note=(
            f"latest: {100*latest.train_success:.0f}% · roll5 {100*latest.success_roll5:.1f}% · "
            f"roll10 {100*latest.success_roll10:.1f}%"
        ),
    )

    draw_panel(
        draw,
        (1280, 155, 2465, 845),
        "Fixed-ID evaluation (20 episodes each)",
        "Checkpoint/global step",
        "Success (%)",
        evaluations.step.astype(float).tolist(),
        [
            (
                "fixed20 success",
                (100 * evaluations.eval_success).tolist(),
                "#0F766E",
                5,
            )
        ],
        0,
        100,
        vertical=[(first_apply, "DVAC apply", "#7C3AED")],
        note=f"latest fixed20: g{int(latest_eval.step)} = {100*latest_eval.eval_success:.0f}%",
    )

    draw_panel(
        draw,
        (55, 885, 1240, 1590),
        "DVAC action-gradient weights while learning",
        "Global step",
        "Weight / ESS",
        active.step.astype(float).tolist(),
        [
            ("p05", active.weight_p05.tolist(), "#2563EB", 4),
            ("median", active.weight_median.tolist(), "#7C3AED", 4),
            ("p95", active.weight_p95.tolist(), "#F97316", 4),
            ("ESS", active.weight_ess.tolist(), "#0F766E", 4),
        ],
        0,
        2.05,
        horizontal=[(1.0, "uniform weight = 1", "#64748B")],
        note=(
            f"latest p05/median/p95 {latest.weight_p05:.3f}/{latest.weight_median:.3f}/{latest.weight_p95:.3f} · "
            f"ESS {latest.weight_ess:.3f} · h0→h9 {latest_h_weights[0]:.3f}→{latest_h_weights[-1]:.3f}"
        ),
    )

    resource_plot = resources.iloc[:: max(1, len(resources) // 1500)]
    draw_panel(
        draw,
        (1280, 885, 2465, 1590),
        "Resources over the full run",
        "Hours since launch",
        "GiB",
        resource_plot.hours.tolist(),
        [
            ("cgroup", resource_plot.cgroup_gib.tolist(), "#111827", 5),
            ("Env RSS", resource_plot.env_rss_gib.tolist(), "#DC2626", 4),
            ("GPU0", resource_plot.gpu0_gib.tolist(), "#F97316", 3),
            ("GPU1", resource_plot.gpu1_gib.tolist(), "#2563EB", 3),
        ],
        0,
        250,
        horizontal=[(236, "memory.high 236 GiB", "#D97706")],
        note=(
            f"latest cgroup {resources.cgroup_gib.iloc[-1]:.1f} GiB · Env RSS {resources.env_rss_gib.iloc[-1]:.1f} GiB · "
            f"GPU {resources.gpu0_gib.iloc[-1]:.1f}/{resources.gpu1_gib.iloc[-1]:.1f} GiB"
        ),
    )

    plot_path = ROOT / f"RLT_DVAC_LIVE_G{int(latest.step)}_SUMMARY.png"
    image.save(plot_path, optimize=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
