from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

from PIL import Image


ROOT = Path(r"C:\Users\86136\Documents\rl")
EVIDENCE = ROOT / "docs" / "rlinf-shenzhen-pi0-ppo-rlt" / "evidence"
SNAPSHOT = EVIDENCE / "dvac-grpo-v4-live-step31-20260825"
BASE_SCRIPT = ROOT / "local_scripts" / "render_shenzhen_grpo_vs_ppo_20260823.py"
AUTODL_CSV = (
    ROOT
    / "docs"
    / "rlinf-robotwin-pi0-dvac-telemetry"
    / "evidence"
    / "global_z_w0to2_stop_g49_20260824"
    / "analysis"
    / "FIVE_RUN_SUCCESS_RAW_ROLL5_ROLL10_G49.csv"
)
SHENZHEN_BASELINE_CSV = EVIDENCE / "grpo_v2_final_step52_20260823" / "grpo_console_scalars_step1_52.csv"
OUT = SNAPSHOT / "04_autodl_vs_shenzhen_global_z_effect_same_machine_controls.png"


def load_base():
    spec = importlib.util.spec_from_file_location("plot_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B = load_base()
ORANGE = "#D97706"
ORANGE_LIGHT = "#FBBF24"


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def padded(values: list[float | None], length: int) -> list[float | None]:
    return values + [None] * (length - len(values))


def split_paired(values: list[float | None], paired: int) -> tuple[list[float | None], list[float | None]]:
    first = [value if index < paired else None for index, value in enumerate(values)]
    future = [None if index < paired - 1 else value for index, value in enumerate(values)]
    return first, future


def cut(draw, plot, x_min: float, x_max: float, value: float, label: str) -> None:
    left, top, right, bottom = plot
    x = left + (value - x_min) / (x_max - x_min) * (right - left)
    for y in range(int(top), int(bottom), 18):
        draw.line((x, y, x, min(y + 9, bottom)), fill=B.GRAY, width=2)
    draw.text((x + 8, top + 12), label, fill=B.GRAY, font=B.F_SMALL)


def main() -> None:
    old = csv_rows(AUTODL_CSV)
    old_dvac = [float(row["success_roll5_pct"]) / 100 for row in old if row["run"] == "DVAC global-z [0,2]"]
    old_base = [float(row["success_roll5_pct"]) / 100 for row in old if row["run"] == "Original GRPO"]
    old_base_paired, old_base_future = split_paired(old_base, len(old_dvac))

    sz_dvac_rows = B.parse_grpo(SNAPSHOT / "driver.log")
    sz_dvac = B.trailing([float(row["train_success"]) for row in sz_dvac_rows])
    sz_base_rows = csv_rows(SHENZHEN_BASELINE_CSV)
    sz_base = B.trailing([float(row["train_success"]) for row in sz_base_rows])
    sz_base_paired, sz_base_future = split_paired(sz_base, len(sz_dvac))

    image = Image.new("RGB", (1440, 1820), B.LIGHT)
    draw = B.add_header(
        image,
        "Global-z [0,2]: AutoDL and Shenzhen within-machine comparisons",
        "Five-step training-rollout means. Each DVAC curve is compared only with its own machine's baseline.",
    )

    old_x = [float(step) for step in range(1, 101)]
    old_plot = B.line_panel(
        draw,
        (45, 145, 1395, 620),
        old_x,
        [
            ("DVAC", padded(old_dvac, 100), ORANGE, 7),
            ("GRPO paired", old_base_paired, B.GRPO, 6),
            ("GRPO post-g49", old_base_future, B.GRPO_LIGHT, 3),
        ],
        0.70,
        1.0,
        "AutoDL (2 A800; 256 traj/step; B512)",
    )
    cut(draw, old_plot, 1.0, 100.0, 49.5, "after g49: baseline only")

    sz_x = [float(step) for step in range(1, 53)]
    sz_plot = B.line_panel(
        draw,
        (45, 650, 1395, 1125),
        sz_x,
        [
            ("DVAC", padded(sz_dvac, 52), B.PURPLE, 7),
            ("GRPO paired", sz_base_paired, B.GRPO, 6),
            ("GRPO post-31", sz_base_future, B.GRPO_LIGHT, 3),
        ],
        0.70,
        1.0,
        "Shenzhen (4 H100; 512 traj/step; B2048)",
    )
    cut(draw, sz_plot, 1.0, 52.0, 31.5, "after Step31: baseline only")

    delta_x = [float(step) for step in range(1, 50)]
    old_delta = [method - base for method, base in zip(old_dvac, old_base[: len(old_dvac)])]
    sz_delta = [
        None if method is None or base is None else float(method) - float(base)
        for method, base in zip(sz_dvac, sz_base[: len(sz_dvac)])
    ]
    delta_plot = B.line_panel(
        draw,
        (45, 1155, 1395, 1640),
        delta_x,
        [
            ("AutoDL delta", old_delta, ORANGE, 7),
            ("Shenzhen delta", padded(sz_delta, 49), B.PURPLE, 7),
        ],
        -0.08,
        0.08,
        "Five-step DVAC minus matched GRPO",
        zero_line=True,
    )
    cut(draw, delta_plot, 1.0, 49.0, 31.5, "Shenzhen observed through Step31")

    draw.rounded_rectangle((55, 1665, 1385, 1800), radius=18, fill=B.WHITE, outline=B.GRID, width=2)
    draw.text(
        (76, 1686),
        "AutoDL g1-49: +2.08 pp cumulative, +2.73 pp latest-5; no fixed held-out eval.",
        fill=B.INK,
        font=B.F_SMALL,
    )
    draw.text(
        (76, 1723),
        "Shenzhen Step1-31: -0.50 pp cumulative, -1.25 pp latest-5; fixed cumulative 179/192 vs 179/192.",
        fill=B.GRAY,
        font=B.F_SMALL,
    )
    draw.text(
        (76, 1760),
        "Cross-machine steps carry different sample budgets; use this as a replication diagnosis, not a pooled causal estimate.",
        fill=B.GRAY,
        font=B.F_SMALL,
    )
    image.save(OUT, optimize=True)
    print(OUT)


if __name__ == "__main__":
    main()
