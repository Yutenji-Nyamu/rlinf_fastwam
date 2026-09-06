from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(
    r"C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-dvac-telemetry"
    r"\evidence\rlt_dvac_formal_live_g65_20260824"
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
        step = int(start.group(1))
        env = section(block, "Environment")
        evaluation = section(block, "Evaluation")
        elapsed_match = re.search(r"Elapsed:\s*([^│]+?)\s*│", block)
        rows.append(
            {
                "step": step,
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
                "baseline_count": first_float(block, "rlt_dvac/baseline_count"),
                "baseline_frozen": first_float(block, "rlt_dvac/baseline_frozen"),
                "weight_mean": first_float(block, "rlt_dvac/weight_mean"),
                "weight_ess": first_float(block, "rlt_dvac/weight_ess"),
            }
        )
    frame = pd.DataFrame(rows).drop_duplicates("step", keep="last").sort_values("step")
    frame["success_roll5"] = frame["train_success"].rolling(5, min_periods=1).mean()
    frame["success_roll10"] = frame["train_success"].rolling(10, min_periods=1).mean()
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
) -> None:
    left, top, right, bottom = rect
    draw.rounded_rectangle(rect, radius=18, fill="#FFFFFF", outline="#D7DEE9", width=2)
    draw.text((left + 28, top + 22), title, fill="#172033", font=font(27, True))
    legend_y = top + 70
    legend_x = left + 30
    for name, _, color, width in series:
        draw.line((legend_x, legend_y + 10, legend_x + 34, legend_y + 10), fill=color, width=max(3, width))
        draw.text((legend_x + 44, legend_y), name, fill="#475569", font=font(17))
        legend_x += 44 + int(draw.textlength(name, font=font(17))) + 28

    plot_left, plot_top = left + 105, top + 118
    plot_right, plot_bottom = right - 32, bottom - 70
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
        label = f"{x:.0f}"
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
                draw.text((plot_right - draw.textlength(label, font=font(15)) - 4, yp - 23), label, fill=color, font=font(15))

    for _, values, color, width in series:
        points = [
            (px(float(x)), py(float(y)))
            for x, y in zip(x_values, values)
            if pd.notna(y) and y_min <= float(y) <= y_max
        ]
        if len(points) >= 2:
            draw.line(points, fill=color, width=width, joint="curve")
        elif len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)

    draw.text(((plot_left + plot_right) / 2 - draw.textlength(x_label, font=font(17)) / 2, bottom - 42), x_label, fill="#475569", font=font(17))
    draw.text((left + 18, plot_top - 30), y_label, fill="#475569", font=font(16))


def parse_resources(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    numeric = [column for column in frame.columns if column != "unix_time"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["unix_time", "cgroup_current_bytes"])
    frame["minutes"] = (frame["unix_time"] - frame["unix_time"].iloc[0]) / 60.0
    gib = 1024.0**3
    frame["cgroup_gib"] = frame["cgroup_current_bytes"] / gib
    frame["anon_gib"] = frame["anon_bytes"] / gib
    frame["file_gib"] = frame["file_bytes"] / gib
    frame["env_rss_gib"] = frame["env_rss_kib"] / 1024.0**2
    frame["gpu0_gib"] = frame["gpu0_used_mib"] / 1024.0
    frame["gpu1_gib"] = frame["gpu1_used_mib"] / 1024.0
    frame["gpu0_util_1m"] = frame["gpu0_util_pct"].rolling(30, min_periods=1).mean()
    frame["gpu1_util_1m"] = frame["gpu1_util_pct"].rolling(30, min_periods=1).mean()
    return frame


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    steps = parse_metrics(RAW / "metrics.log")
    resources = parse_resources(RAW / "resources.csv")
    steps.to_csv(ROOT / "step_metrics.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    latest = steps.iloc[-1]
    summary = {
        "latest_complete_step": int(latest.step),
        "progress_fraction": float(latest.step / 480.0),
        "latest_train_success": float(latest.train_success),
        "latest_success_roll5": float(latest.success_roll5),
        "latest_success_roll10": float(latest.success_roll10),
        "cumulative_train_success": float(steps.train_success.mean()),
        "eval_points": {
            str(int(row.step)): float(row.eval_success)
            for _, row in steps.dropna(subset=["eval_success"]).iterrows()
        },
        "global_min_replay": float(latest.global_min_replay),
        "replay_readiness_fraction": float(latest.global_min_replay / 10000.0),
        "actor_updates": float(latest.actor_updates),
        "critic_updates": float(latest.critic_updates),
        "baseline_count": float(latest.baseline_count),
        "baseline_frozen": bool(latest.baseline_frozen),
        "cgroup_current_gib": float(resources.cgroup_gib.iloc[-1]),
        "cgroup_peak_gib": float(resources.cgroup_gib.max()),
        "anon_current_gib": float(resources.anon_gib.iloc[-1]),
        "file_current_gib": float(resources.file_gib.iloc[-1]),
        "env_rss_current_gib": float(resources.env_rss_gib.iloc[-1]),
        "env_rss_peak_gib": float(resources.env_rss_gib.max()),
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

    image = Image.new("RGB", (2520, 1620), "#F5F7FB")
    draw = ImageDraw.Draw(image)
    draw.text((70, 42), f"RLT teacher-DVAC fresh-480 — live through completed step {int(latest.step)}", fill="#172033", font=font(42, True))
    draw.text((72, 98), "Current phase: replay collection / warmup; policy updates and DVAC weighting have not started.", fill="#58657A", font=font(23))

    draw_panel(
        draw,
        (55, 155, 1240, 830),
        "Rollout success: policy has not started updating",
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
    )
    draw_panel(
        draw,
        (1280, 155, 2465, 830),
        "Replay warmup: DVAC weights not applied yet",
        "Global step",
        "Transitions / rank",
        steps.step.astype(float).tolist(),
        [("min replay across ranks", steps.global_min_replay.tolist(), "#0F766E", 5)],
        0,
        11000,
        [(10000, "online readiness = 10,000", "#DC2626")],
    )

    res_plot = resources.iloc[:: max(1, len(resources) // 1200)]
    draw_panel(
        draw,
        (55, 870, 1240, 1545),
        "Host memory remains far below cgroup pressure",
        "Minutes since launch",
        "GiB",
        res_plot.minutes.tolist(),
        [
            ("cgroup current", res_plot.cgroup_gib.tolist(), "#111827", 5),
            ("anonymous", res_plot.anon_gib.tolist(), "#DC2626", 4),
            ("file/cache", res_plot.file_gib.tolist(), "#2563EB", 4),
        ],
        0,
        250,
        [(236, "memory.high 236 GiB", "#F59E0B")],
    )
    draw_panel(
        draw,
        (1280, 870, 2465, 1545),
        "Two-GPU memory",
        "Minutes since launch",
        "GiB",
        res_plot.minutes.tolist(),
        [
            ("GPU0 memory", res_plot.gpu0_gib.tolist(), "#F97316", 4),
            ("GPU1 memory", res_plot.gpu1_gib.tolist(), "#2563EB", 4),
        ],
        0,
        80,
    )
    image.save(ROOT / "RLT_DVAC_LIVE_G66_SUMMARY.png", optimize=True)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
