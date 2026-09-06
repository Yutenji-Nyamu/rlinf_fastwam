from __future__ import annotations

import csv
import importlib.util
import json
import statistics
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(r"C:\Users\86136\Documents\rl")
EVIDENCE = ROOT / "docs" / "rlinf-shenzhen-pi0-ppo-rlt" / "evidence"
SNAPSHOT = EVIDENCE / "dual-2gpu-comparison-live-20260827"
BASE_SCRIPT = ROOT / "local_scripts" / "render_shenzhen_grpo_vs_ppo_20260823.py"
OLD_SZ_CSV = EVIDENCE / "grpo_v2_final_step52_20260823" / "grpo_console_scalars_step1_52.csv"
OLD_SZ_DVAC_CSV = (
    EVIDENCE
    / "dvac-grpo-w0to2-final-step41-20260825"
    / "dvac_actual_branch_console_step1_41.csv"
)
AUTODL_CSV = (
    ROOT
    / "docs"
    / "rlinf-robotwin-pi0-dvac-telemetry"
    / "evidence"
    / "global_z_w0to2_stop_g49_20260824"
    / "analysis"
    / "FIVE_RUN_SUCCESS_RAW_ROLL5_ROLL10_G49.csv"
)

CURRENT_OUT = SNAPSHOT / "01_current_2gpu_control_vs_dvac_raw_roll5_roll10.png"
CONTEXT_OUT = SNAPSHOT / "02_shenzhen_old_grpo_and_autodl_context.png"
CSV_OUT = SNAPSHOT / "current_2gpu_success_curves.csv"
SUMMARY_OUT = SNAPSHOT / "comparison_summary.json"

DARK_TEAL = "#0F766E"
TEAL_LIGHT = "#5EEAD4"
ORANGE = "#D97706"
ORANGE_LIGHT = "#FBBF24"
SLATE = "#64748B"
SLATE_LIGHT = "#CBD5E1"


def load_base():
    spec = importlib.util.spec_from_file_location("plot_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B = load_base()


def trailing(values: list[float], window: int) -> list[float]:
    return [statistics.mean(values[max(0, idx - window + 1) : idx + 1]) for idx in range(len(values))]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def overlay_markers(
    draw: ImageDraw.ImageDraw,
    plot: tuple[float, float, float, float],
    xs: list[float],
    ys: list[float | None],
    y_min: float,
    y_max: float,
    color: str,
    shape: str,
    radius: int = 4,
) -> None:
    left, top, right, bottom = plot
    x_min, x_max = min(xs), max(xs)
    for x_value, y_value in zip(xs, ys):
        if y_value is None:
            continue
        x = left + (x_value - x_min) / max(1e-9, x_max - x_min) * (right - left)
        y = bottom - (float(y_value) - y_min) / (y_max - y_min) * (bottom - top)
        if shape == "square":
            draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=color)
        elif shape == "triangle":
            draw.polygon(((x, y - radius - 1), (x - radius, y + radius), (x + radius, y + radius)), fill=color)
        else:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def run_series() -> tuple[list[dict], list[dict]]:
    control = B.parse_grpo(SNAPSHOT / "raw" / "control" / "runtime" / "driver.log")
    dvac = B.parse_grpo(SNAPSHOT / "raw" / "dvac_w0to2" / "runtime" / "driver.log")
    common = min(len(control), len(dvac))
    return control[:common], dvac[:common]


def write_current_csv(control: list[dict], dvac: list[dict]) -> None:
    control_raw = [float(row["train_success"]) for row in control]
    dvac_raw = [float(row["train_success"]) for row in dvac]
    control5, control10 = trailing(control_raw, 5), trailing(control_raw, 10)
    dvac5, dvac10 = trailing(dvac_raw, 5), trailing(dvac_raw, 10)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "step", "control_raw", "dvac_raw", "control_roll5", "dvac_roll5",
                "control_roll10", "dvac_roll10", "control_eval", "dvac_eval",
            ),
        )
        writer.writeheader()
        for idx, (control_row, dvac_row) in enumerate(zip(control, dvac)):
            writer.writerow(
                {
                    "step": idx + 1,
                    "control_raw": control_raw[idx],
                    "dvac_raw": dvac_raw[idx],
                    "control_roll5": control5[idx],
                    "dvac_roll5": dvac5[idx],
                    "control_roll10": control10[idx],
                    "dvac_roll10": dvac10[idx],
                    "control_eval": control_row["eval_success"] if control_row["eval_success"] is not None else "",
                    "dvac_eval": dvac_row["eval_success"] if dvac_row["eval_success"] is not None else "",
                }
            )


def render_current(control: list[dict], dvac: list[dict]) -> None:
    steps = [float(row["step"]) for row in control]
    control_raw = [float(row["train_success"]) for row in control]
    dvac_raw = [float(row["train_success"]) for row in dvac]
    control5, control10 = trailing(control_raw, 5), trailing(control_raw, 10)
    dvac5, dvac10 = trailing(dvac_raw, 5), trailing(dvac_raw, 10)
    eval_steps = {int(row["step"]) for row in control if row["eval_success"] is not None}

    image = Image.new("RGB", (1440, 1880), B.LIGHT)
    draw = B.add_header(
        image,
        f"SZ two-GPU matched pair through Step {len(steps)}",
        "Control uses dark teal circles; DVAC [0,2] uses orange squares. Training budget is identical.",
    )
    panels = (
        ("Raw training-rollout success", control_raw, dvac_raw),
        ("Five-step trailing mean", control5, dvac5),
        ("Ten-step trailing mean", control10, dvac10),
    )
    top, height, gap = 145, 490, 32
    for idx, (title, control_values, dvac_values) in enumerate(panels):
        box = (45, top + idx * (height + gap), 1395, top + idx * (height + gap) + height)
        plot = B.line_panel(
            draw,
            box,
            steps,
            [
                ("Control", control_values, DARK_TEAL, 6),
                ("DVAC [0,2]", dvac_values, ORANGE, 6),
            ],
            0.68,
            1.0,
            title,
            eval_steps=eval_steps,
        )
        overlay_markers(draw, plot, steps, control_values, 0.68, 1.0, DARK_TEAL, "circle", 4)
        overlay_markers(draw, plot, steps, dvac_values, 0.68, 1.0, ORANGE, "square", 4)
        if idx == 0:
            B.draw_eval_markers(
                draw,
                plot,
                steps,
                0.68,
                1.0,
                [(int(row["step"]), row["eval_success"]) for row in control],
                DARK_TEAL,
                "square",
            )
            B.draw_eval_markers(
                draw,
                plot,
                steps,
                0.68,
                1.0,
                [(int(row["step"]), row["eval_success"]) for row in dvac],
                ORANGE,
                "triangle",
            )
            draw.text(
                (84, box[3] - 42),
                "Large square/triangle markers are paired fixed-32 evaluations; shaded columns are eval steps.",
                fill=B.GRAY,
                font=B.F_SMALL,
            )

    mean_delta = (statistics.mean(dvac_raw) - statistics.mean(control_raw)) * 100
    last5_delta = (dvac5[-1] - control5[-1]) * 100
    last10_delta = (dvac10[-1] - control10[-1]) * 100
    draw.rounded_rectangle((55, 1720, 1385, 1850), radius=18, fill=B.WHITE, outline=B.GRID, width=2)
    draw.text(
        (78, 1744),
        f"DVAC-Control through Step {len(steps)}: mean {mean_delta:+.2f} pp; latest-5 {last5_delta:+.2f} pp; latest-10 {last10_delta:+.2f} pp.",
        fill=B.INK,
        font=B.F_CARD,
    )
    draw.text(
        (78, 1793),
        "These are noisy on-policy training rollouts; fixed evaluation points carry more weight than a single raw step.",
        fill=B.GRAY,
        font=B.F_SMALL,
    )
    image.save(CURRENT_OUT, optimize=True)


def render_context(control: list[dict]) -> None:
    old_rows = read_csv(OLD_SZ_CSV)
    old_raw = [float(row["train_success"]) for row in old_rows]
    old5, old10 = trailing(old_raw, 5), trailing(old_raw, 10)
    old_dvac_rows = read_csv(OLD_SZ_DVAC_CSV)
    old_dvac_raw = [float(row["train_success"]) for row in old_dvac_rows]
    old_dvac5, old_dvac10 = trailing(old_dvac_raw, 5), trailing(old_dvac_raw, 10)

    autodl_rows = read_csv(AUTODL_CSV)
    autodl_base = [row for row in autodl_rows if row["run"] == "Original GRPO"]
    autodl_dvac = [row for row in autodl_rows if row["run"] == "DVAC global-z [0,2]"]
    base5 = [float(row["success_roll5_pct"]) / 100 for row in autodl_base]
    base10 = [float(row["success_roll10_pct"]) / 100 for row in autodl_base]
    method5 = [float(row["success_roll5_pct"]) / 100 for row in autodl_dvac]
    method10 = [float(row["success_roll10_pct"]) / 100 for row in autodl_dvac]
    base_raw = [float(row["success_pct"]) / 100 for row in autodl_base]
    method_raw = [float(row["success_pct"]) / 100 for row in autodl_dvac]

    image = Image.new("RGB", (1440, 3260), B.LIGHT)
    draw = B.add_header(
        image,
        "Historical within-machine pairs",
        "Each panel compares Control with DVAC [0,2] only inside one machine/configuration.",
    )
    specs = (
        (
            "Previous Shenzhen 4-GPU pair: raw success",
            list(map(float, range(1, 53))),
            [
                ("GRPO", old_raw, DARK_TEAL, 6),
                ("DVAC [0,2]", old_dvac_raw + [None] * (52 - len(old_dvac_raw)), ORANGE, 6),
            ],
        ),
        (
            "Previous Shenzhen 4-GPU pair: five-step mean",
            list(map(float, range(1, 53))),
            [
                ("GRPO", old5, DARK_TEAL, 6),
                ("DVAC [0,2]", old_dvac5 + [None] * (52 - len(old_dvac5)), ORANGE, 6),
            ],
        ),
        (
            "Previous Shenzhen 4-GPU pair: ten-step mean",
            list(map(float, range(1, 53))),
            [
                ("GRPO", old10, DARK_TEAL, 6),
                ("DVAC [0,2]", old_dvac10 + [None] * (52 - len(old_dvac10)), ORANGE, 6),
            ],
        ),
        (
            "AutoDL pair: raw success",
            list(map(float, range(1, 101))),
            [
                ("GRPO", base_raw, DARK_TEAL, 6),
                ("DVAC [0,2]", method_raw + [None] * (100 - len(method_raw)), ORANGE, 6),
            ],
        ),
        (
            "AutoDL pair: five-step mean",
            list(map(float, range(1, 101))),
            [
                ("GRPO", base5, DARK_TEAL, 6),
                ("DVAC [0,2]", method5 + [None] * (100 - len(method5)), ORANGE, 6),
            ],
        ),
        (
            "AutoDL pair: ten-step mean",
            list(map(float, range(1, 101))),
            [
                ("GRPO", base10, DARK_TEAL, 6),
                ("DVAC [0,2]", method10 + [None] * (100 - len(method10)), ORANGE, 6),
            ],
        ),
    )
    top, height, gap = 145, 450, 24
    for idx, (title, xs, series) in enumerate(specs):
        plot = B.line_panel(
            draw,
            (45, top + idx * (height + gap), 1395, top + idx * (height + gap) + height),
            xs,
            series,
            0.68,
            1.0,
            title,
        )
        for series_idx, (_, values, color, _) in enumerate(series):
            overlay_markers(draw, plot, xs, values, 0.68, 1.0, color, "circle" if series_idx == 0 else "square", 4)

    draw.rounded_rectangle((55, 3025, 1385, 3235), radius=18, fill=B.WHITE, outline=B.GRID, width=2)
    draw.text(
        (76, 3046),
        "Previous SZ Step1-41 DVAC-Control: mean -0.39 pp; latest-5 -0.66 pp; latest-10 -0.33 pp; fixed64 tied 242/256.",
        fill=B.INK,
        font=B.F_SMALL,
    )
    draw.text(
        (76, 3088),
        "AutoDL Step1-49 DVAC-Control: mean +2.08 pp; latest-5 +2.73 pp; latest-10 +2.93 pp; no held-out evaluation.",
        fill=B.INK,
        font=B.F_SMALL,
    )
    draw.text(
        (76, 3130),
        "Budgets: previous SZ=512 trajectories/step, B2048; AutoDL=256 trajectories/step, B512. Cross-panel levels are not causal.",
        fill=B.GRAY,
        font=B.F_SMALL,
    )
    draw.text(
        (76, 3172),
        "AutoDL also lacked same-seed same-code mode-off and repeated seeds; its positive gap may include run variance.",
        fill=B.GRAY,
        font=B.F_SMALL,
    )
    image.save(CONTEXT_OUT, optimize=True)


def build_summary(control: list[dict], dvac: list[dict]) -> dict:
    control_raw = [float(row["train_success"]) for row in control]
    dvac_raw = [float(row["train_success"]) for row in dvac]
    control5, control10 = trailing(control_raw, 5), trailing(control_raw, 10)
    dvac5, dvac10 = trailing(dvac_raw, 5), trailing(dvac_raw, 10)
    control_eval = [float(row["eval_success"]) for row in control if row["eval_success"] is not None]
    dvac_eval = [float(row["eval_success"]) for row in dvac if row["eval_success"] is not None]

    autodl_rows = read_csv(AUTODL_CSV)
    base = [row for row in autodl_rows if row["run"] == "Original GRPO"]
    method = [row for row in autodl_rows if row["run"] == "DVAC global-z [0,2]"]
    common = min(len(base), len(method))
    base_raw = [float(row["success_pct"]) for row in base[:common]]
    method_raw = [float(row["success_pct"]) for row in method[:common]]

    return {
        "snapshot_complete_steps": len(control),
        "smoothing": "trailing arithmetic mean; min_periods=1; no forward fill",
        "current_two_gpu": {
            "control_mean_pct": 100 * statistics.mean(control_raw),
            "dvac_mean_pct": 100 * statistics.mean(dvac_raw),
            "dvac_minus_control_mean_pp": 100 * (statistics.mean(dvac_raw) - statistics.mean(control_raw)),
            "dvac_minus_control_latest5_pp": 100 * (dvac5[-1] - control5[-1]),
            "dvac_minus_control_latest10_pp": 100 * (dvac10[-1] - control10[-1]),
            "control_fixed32_successes": round(32 * sum(control_eval)),
            "dvac_fixed32_successes": round(32 * sum(dvac_eval)),
            "fixed32_episodes_each": 32 * len(control_eval),
        },
        "autodl_common_g1_to_g49": {
            "dvac_minus_control_mean_pp": statistics.mean(method_raw) - statistics.mean(base_raw),
            "dvac_minus_control_latest5_pp": float(method[common - 1]["success_roll5_pct"]) - float(base[common - 1]["success_roll5_pct"]),
            "dvac_minus_control_latest10_pp": float(method[common - 1]["success_roll10_pct"]) - float(base[common - 1]["success_roll10_pct"]),
        },
    }


def main() -> None:
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    control, dvac = run_series()
    write_current_csv(control, dvac)
    render_current(control, dvac)
    render_context(control)
    SUMMARY_OUT.write_text(json.dumps(build_summary(control, dvac), indent=2) + "\n", encoding="utf-8")
    for path in (CURRENT_OUT, CONTEXT_OUT, CSV_OUT, SUMMARY_OUT):
        print(path)


if __name__ == "__main__":
    main()
