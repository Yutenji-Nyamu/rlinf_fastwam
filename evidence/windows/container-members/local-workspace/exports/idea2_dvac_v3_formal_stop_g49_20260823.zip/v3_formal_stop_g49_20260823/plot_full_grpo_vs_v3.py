from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
ORIGINAL = (
    WORKSPACE
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/"
    "three_run_training_comparison_20260822/THREE_RUN_METRICS_G23.csv"
)
FOUR_RUN = HERE / "analysis/FOUR_RUN_TRAINING_G49.csv"
OUT = HERE / "analysis"
PLOT_HELPER = (
    WORKSPACE
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/"
    "r_only_formal_live_g50_20260822/analyze_live_g50.py"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    original_all = pd.read_csv(ORIGINAL)
    original = original_all[
        original_all["run"].astype(str).str.startswith("Original GRPO")
    ].copy()
    original = original.drop_duplicates("step").sort_values("step")
    v3_all = pd.read_csv(FOUR_RUN)
    v3 = v3_all[v3_all.run == "DVAC v3 R-only [0,2]"].copy().sort_values("step")

    original["success_5step"] = original.success_pct.rolling(5, min_periods=1).mean()
    v3["success_5step"] = v3.success_pct.rolling(5, min_periods=1).mean()
    original["series"] = "Original GRPO"
    v3["series"] = "DVAC v3 R-only [0,2]"
    compact = pd.concat(
        [
            original[["series", "step", "success_pct", "success_5step"]],
            v3[["series", "step", "success_pct", "success_5step"]],
        ],
        ignore_index=True,
    )
    compact.to_csv(OUT / "ORIGINAL_GRPO_FULL100_VS_V3_G49_SUCCESS.csv", index=False)

    plot = load_module("full_grpo_v3_plot", PLOT_HELPER)
    image, draw = plot.make_canvas(
        "Original GRPO full 100-step success vs DVAC v3 through g49",
        "Raw per-step rollout success and trailing 5-step means. This is on-policy training data, not held-out evaluation.",
        1040,
    )
    raw_box = (55, 150, 1745, 555)
    mean_box = (55, 615, 1745, 1005)
    plot.draw_chart(
        draw,
        raw_box,
        "A. Every training step",
        [
            ("Original GRPO — full 100", original.step.to_numpy(float), original.success_pct.to_numpy(float), plot.BLUE),
            ("DVAC v3 — stopped at g49", v3.step.to_numpy(float), v3.success_pct.to_numpy(float), plot.ORANGE),
        ],
        "success (%)",
        x_limits=(1, 100),
        legend=True,
    )
    plot.draw_chart(
        draw,
        mean_box,
        "B. Trailing 5-step mean",
        [
            ("Original GRPO — full 100", original.step.to_numpy(float), original.success_5step.to_numpy(float), plot.BLUE),
            ("DVAC v3 — stopped at g49", v3.step.to_numpy(float), v3.success_5step.to_numpy(float), plot.ORANGE),
        ],
        "success (%)",
        x_limits=(1, 100),
        legend=True,
    )
    image.save(OUT / "ORIGINAL_GRPO_FULL100_VS_V3_G49_SUCCESS.png", optimize=True)


if __name__ == "__main__":
    main()
