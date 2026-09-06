from __future__ import annotations

import importlib.util
import json
import math
import statistics
from datetime import timezone, timedelta
from pathlib import Path

from PIL import Image


ROOT = Path(r"C:\Users\86136\Documents\rl")
EVIDENCE = ROOT / "docs" / "rlinf-shenzhen-pi0-ppo-rlt" / "evidence"
SNAPSHOT = EVIDENCE / "grpo_v2_live_current_20260823"
LIVE36_SCRIPT = ROOT / "local_scripts" / "render_shenzhen_grpo_live_step36_20260823.py"
PPO_CSV = EVIDENCE / "ppo_formal100_console_scalars_step1_46_20260822.csv"
PPO_RESOURCE_CSV = EVIDENCE / "ppo_formal100_resource_discrete_samples_through_step46_20260822.csv"


def load_live36():
    spec = importlib.util.spec_from_file_location("grpo_live36", LIVE36_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {LIVE36_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


L = load_live36()
B = L.B


def render_resources(resources: list[dict], aligned: list[dict], ppo_resource: list[dict], latest_step: int, output: Path) -> None:
    start = resources[0]["timestamp"]
    elapsed_h = [(row["timestamp"] - start).total_seconds() / 3600 for row in resources]
    last_h = elapsed_h[-1]
    hour_ticks = list(range(0, math.ceil(last_h) + 1, 2))
    if last_h not in hour_ticks:
        hour_ticks.append(last_h)

    img = Image.new("RGB", (1440, 2480), B.LIGHT)
    capture_cst = resources[-1]["timestamp"].astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M CST")
    draw = B.add_header(
        img,
        f"GRPO resource timeline through Step {latest_step}",
        f"One-minute samples through {capture_cst}; GPU utilization is bursty during simulation",
    )
    panels = [
        (
            "Main-memory pressure",
            [
                ("GRPO cgroup used", [float(row["cgroup_gib"]) for row in resources], B.PURPLE, 5),
                ("Host available", [float(row["host_available_gib"]) for row in resources], B.GREEN, 5),
            ],
            0,
            2100,
        ),
        (
            "Physical GPUs 4-7 memory",
            [
                ("Mean GPU memory", [float(row["gpu_mean_gib"]) for row in resources], B.GRPO, 5),
                ("Max card memory", [float(row["gpu_max_gib"]) for row in resources], B.RED, 5),
            ],
            0,
            80,
        ),
        (
            "Physical GPUs 4-7 utilization (one-minute samples)",
            [
                ("Raw mean", [float(row["gpu_mean_util"]) for row in resources], B.GRPO_LIGHT, 2),
                ("15-min mean", L.trailing([float(row["gpu_mean_util"]) for row in resources], 15), B.GRPO, 6),
                ("15-min max-card mean", L.trailing([float(row["gpu_max_util"]) for row in resources], 15), B.RED, 4),
            ],
            0,
            100,
        ),
    ]
    top, height, gap = 145, 500, 25
    for idx, (title, series, low, high) in enumerate(panels):
        B.line_panel(
            draw,
            (45, top + idx * (height + gap), 1395, top + idx * (height + gap) + height),
            elapsed_h,
            series,
            low,
            high,
            title,
            x_label="Hours since observer start",
            x_ticks=hour_ticks,
            x_tick_format=lambda value: f"{value:.1f}" if value == last_h else f"{value:.0f}",
        )

    grpo_steps = [float(row["step"]) for row in aligned]
    grpo_map = {float(row["step"]): float(row["cgroup_gib"]) for row in aligned}
    ppo_boundary = [row for row in ppo_resource if bool(row["boundary"]) and int(row["step"]) <= latest_step]
    ppo_map = {float(row["step"]): float(row["cgroup_gib"]) for row in ppo_boundary}
    combined_x = sorted({*grpo_map, *ppo_map})
    step_ticks = sorted({1, latest_step, *range(5, latest_step + 1, 5)})
    B.line_panel(
        draw,
        (45, 1720, 1395, 2220),
        combined_x,
        [
            ("GRPO nearest-minute", [grpo_map.get(x) for x in combined_x], B.GRPO, 5),
            ("PPO read-only boundaries", [ppo_map.get(x) for x in combined_x], B.PPO, 6),
        ],
        0,
        2000,
        "Cgroup memory by completed step",
        x_ticks=step_ticks,
    )

    latest = resources[-1]
    draw.rounded_rectangle((45, 2250, 1395, 2425), radius=20, fill=B.WHITE, outline=B.GRID, width=2)
    draw.text((69, 2270), "Latest captured observer row", fill=B.INK, font=B.F_PANEL)
    cards = [
        ("Cgroup", f"{latest['cgroup_gib']:.1f} GiB"),
        ("Host available", f"{latest['host_available_gib']:.1f} GiB"),
        ("GPU mean", f"{latest['gpu_mean_gib']:.1f} GiB/card"),
        ("GPU util mean", f"{latest['gpu_mean_util']:.1f}%"),
    ]
    for idx, (label, value) in enumerate(cards):
        x = 80 + idx * 323
        draw.text((x, 2332), label, fill=B.GRAY, font=B.F_SMALL)
        draw.text((x, 2362), value, fill=B.INK, font=B.F_CARD)
    img.save(output, optimize=True)


def slope(rows: list[dict], start: int, end: int, key: str) -> float | None:
    selected = [row for row in rows if start <= int(row["step"]) <= end]
    if len(selected) < 2:
        return None
    return L.linear_slope_per_step(rows, start, end, key)


def main() -> None:
    grpo = B.parse_grpo(SNAPSHOT / "metrics.log")
    latest_step = int(grpo[-1]["step"])
    expected = list(range(1, latest_step + 1))
    if [int(row["step"]) for row in grpo] != expected:
        raise ValueError("GRPO steps are not consecutive from 1")
    if latest_step > 46:
        raise ValueError("PPO comparison snapshot only covers Step 1-46")

    ppo = B.read_ppo(PPO_CSV, latest_step)
    resources = L.read_resources(SNAPSHOT / "resource.csv")
    aligned = L.align_resources(grpo, resources)
    ppo_resource = B.read_ppo_resource(PPO_RESOURCE_CSV)

    scalars_out = SNAPSHOT / f"grpo_console_scalars_step1_{latest_step}.csv"
    resource_step_out = SNAPSHOT / f"grpo_resource_nearest_minute_step1_{latest_step}.csv"
    success_out = SNAPSHOT / f"grpo_vs_ppo_success_step1_{latest_step}.png"
    optimization_out = SNAPSHOT / f"grpo_vs_ppo_optimization_timing_step1_{latest_step}.png"
    resource_out = SNAPSHOT / f"grpo_resource_timeline_through_step{latest_step}.png"
    summary_out = SNAPSHOT / f"summary_step{latest_step}.json"

    B.write_rows(scalars_out, grpo)
    B.write_rows(resource_step_out, aligned)

    original_add_header = B.add_header

    def current_header(img: Image.Image, title: str, subtitle: str):
        return original_add_header(img, title.replace("Step 36", f"Step {latest_step}"), subtitle)

    B.add_header = current_header
    L.SUCCESS_OUT = success_out
    L.OPT_OUT = optimization_out
    L.render_success(grpo, ppo)
    L.render_optimization(grpo, ppo)
    render_resources(resources, aligned, ppo_resource, latest_step, resource_out)
    B.add_header = original_add_header

    L.SCALARS_OUT = scalars_out
    L.RESOURCE_STEP_OUT = resource_step_out
    L.SUCCESS_OUT = success_out
    L.OPT_OUT = optimization_out
    L.RESOURCE_OUT = resource_out
    summary = L.summarize(grpo, ppo, resources, aligned, ppo_resource)
    summary["resources"]["cgroup_slopes_gib_per_step"] = {
        "step_21_30": slope(aligned, 21, 30, "cgroup_gib"),
        "step_31_36": slope(aligned, 31, 36, "cgroup_gib"),
        f"step_37_{latest_step}": slope(aligned, 37, latest_step, "cgroup_gib"),
    }
    summary["timing"]["rollout_fraction_of_median_step"] = (
        float(summary["timing"]["median_rollout_s"]) / float(summary["timing"]["median_step_s"])
    )
    summary["capture"]["complete_step_sequence_verified"] = True
    summary["outputs"] = [str(path) for path in (scalars_out, resource_step_out, success_out, optimization_out, resource_out)]
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

