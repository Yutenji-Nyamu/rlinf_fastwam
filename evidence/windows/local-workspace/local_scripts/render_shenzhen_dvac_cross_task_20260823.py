from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = {
    "episode_mean_S_std": "S (query-wide)",
    "episode_mean_abs_I_std": "|I| (within-chunk)",
    "episode_mean_abs_R_std": "|R| (combined residual)",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.analysis_root.resolve()

    outcome = pd.read_csv(root / "outcome_summary.csv")
    outcome = outcome[
        outcome["metric"].isin(METRICS)
        & (outcome["failure_episodes"] > 0)
    ].copy()
    outcome["label"] = np.where(
        outcome["policy"].eq("pi0"),
        "pi0 / adjust",
        "FW / " + outcome["task"].str.replace("_", " ", regex=False),
    )
    order = [
        "pi0 / adjust",
        "FW / move stapler pad",
        "FW / turn switch",
        "FW / pick diverse bottles",
    ]

    horizon = pd.read_csv(
        root / "query_horizon.csv",
        usecols=["policy", "task", "group_id", "L", "h", "b_position"],
    )
    position = (
        horizon[horizon["L"].eq(3)]
        .drop_duplicates(["group_id", "h"])
        .groupby(["policy", "task"], as_index=False)
        .agg(b_min=("b_position", "min"), b_max=("b_position", "max"))
    )
    position["range"] = position["b_max"] - position["b_min"]
    position["label"] = np.where(
        position["policy"].eq("pi0"),
        "pi0\nadjust",
        "FW\n" + position["task"].str.replace("_", "\n", regex=False),
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)

    x = np.arange(len(order), dtype=float)
    offsets = [-0.22, 0.0, 0.22]
    colors = ["#3366b0", "#e18a2b", "#6f6f6f"]
    for offset, (metric, legend), color in zip(offsets, METRICS.items(), colors):
        frame = outcome[outcome["metric"].eq(metric)].set_index("label").reindex(order)
        center = frame["success_minus_failure"].to_numpy(float)
        low = frame["bootstrap_ci_low"].to_numpy(float)
        high = frame["bootstrap_ci_high"].to_numpy(float)
        axes[0].errorbar(
            x + offset,
            center,
            yerr=np.vstack([center - low, high - center]),
            fmt="o",
            capsize=4,
            linewidth=1.8,
            markersize=6,
            color=color,
            label=legend,
        )
    axes[0].axhline(0, color="#555555", linewidth=1)
    axes[0].set_xticks(x, [value.replace(" / ", "\n") for value in order])
    axes[0].set_ylabel("success - failure (episode mean, 95% bootstrap CI)")
    axes[0].set_title("Outcome association is strongest for S, but its sign is task-dependent")
    axes[0].legend(frameon=False, ncols=3, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    axes[0].grid(axis="y", alpha=0.22)

    position = position.sort_values(["policy", "task"], kind="stable")
    bars = axes[1].bar(
        np.arange(len(position)),
        position["range"],
        color=["#3366b0" if p == "pi0" else "#4c9a73" for p in position["policy"]],
    )
    axes[1].set_xticks(np.arange(len(position)), position["label"])
    axes[1].set_ylabel("max_h b_h - min_h b_h (log units, L=3)")
    axes[1].set_title("Fixed future-position structure is material in every run")
    axes[1].grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, position["range"]):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.035,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle("Shenzhen real rollouts: standardized DVAC decomposition", fontsize=15)
    output = root / "figures" / "cross_task_outcome_and_position_summary.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
