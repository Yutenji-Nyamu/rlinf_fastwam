from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parent
ANALYSIS = json.loads((ROOT / "analysis.json").read_text(encoding="utf-8"))
METRICS = pd.DataFrame(ANALYSIS["metrics"]).sort_values("step")
RESOURCES = pd.read_csv(ROOT / "resources.csv", parse_dates=["timestamp"])

COLORS = {
    "blue": "#2563eb",
    "orange": "#ea580c",
    "green": "#059669",
    "purple": "#7c3aed",
    "red": "#dc2626",
    "gray": "#64748b",
}


def finish(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(ROOT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_success() -> None:
    m = METRICS.copy()
    m["ma5"] = m["success_once"].rolling(5, min_periods=1).mean()
    m["ma10"] = m["success_once"].rolling(10, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(m.step, m.success_once, color=COLORS["gray"], alpha=0.45, linewidth=1,
            marker="o", markersize=2.8, label="Per-step success (64 trajectories)")
    ax.plot(m.step, m.ma5, color=COLORS["orange"], linewidth=2, label="5-step mean")
    ax.plot(m.step, m.ma10, color=COLORS["blue"], linewidth=2.5, label="10-step mean")
    ax.scatter([m.step.iloc[-1]], [m.ma10.iloc[-1]], color=COLORS["blue"], s=36, zorder=4)
    ax.annotate(f"latest 10-step: {m.ma10.iloc[-1]:.1%}",
                (m.step.iloc[-1], m.ma10.iloc[-1]), xytext=(-8, 12),
                textcoords="offset points", ha="right", fontsize=9)
    ax.set(xlabel="RL step", ylabel="Success rate", xlim=(0, m.step.max() + 1), ylim=(0, 1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(True, alpha=0.2)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Fast-WAM GRPO — move_stapler_pad success history")
    finish(fig, "success-history.png")


def plot_optimization() -> None:
    m = METRICS
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.6), sharex=True)

    axes[0].plot(m.step, m["actor/ratio"], color=COLORS["blue"], linewidth=1.6, label="ratio")
    axes[0].axhline(1.0, color=COLORS["gray"], linewidth=1, linestyle="--", label="target 1")
    axes[0].set(ylabel="Ratio", ylim=(0.5, 1.05))
    axes[0].legend(frameon=False, ncol=2, loc="lower right")

    axes[1].plot(m.step, m["actor/approx_kl"], color=COLORS["purple"], linewidth=1.4, label="approx KL")
    axes[1].plot(m.step, m["actor/clip_fraction"], color=COLORS["orange"], linewidth=1.4, label="clip fraction")
    axes[1].axhline(0, color=COLORS["gray"], linewidth=0.8)
    axes[1].set(ylabel="KL / fraction")
    axes[1].legend(frameon=False, ncol=2, loc="upper right")

    axes[2].plot(m.step, m["actor/grad_norm"], color=COLORS["green"], linewidth=1.5, label="grad norm")
    axes[2].set(xlabel="RL step", ylabel="Gradient norm", xlim=(0, m.step.max() + 1))
    axes[2].legend(frameon=False, loc="upper right")

    for ax in axes:
        ax.grid(True, alpha=0.2)
    axes[0].set_title("Fast-WAM GRPO — optimization history")
    finish(fig, "optimization-history.png")


def plot_resources() -> None:
    r = RESOURCES.set_index("timestamp").sort_index()
    one_min = r.resample("1min").max(numeric_only=True)
    elapsed_h = (one_min.index - r.index.min()).total_seconds() / 3600
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.6), sharex=True)

    axes[0].plot(elapsed_h, one_min.cgroup_ram_pct, color=COLORS["red"], linewidth=1.4)
    axes[0].axhline(100, color=COLORS["gray"], linestyle="--", linewidth=1)
    axes[0].set(ylabel="cgroup RAM (%)", ylim=(0, 105))

    axes[1].plot(elapsed_h, one_min.gpu0_memory_mb / 1024, color=COLORS["blue"], linewidth=1.3, label="GPU 0")
    axes[1].plot(elapsed_h, one_min.gpu1_memory_mb / 1024, color=COLORS["orange"], linewidth=1.3, label="GPU 1")
    axes[1].axhline(80, color=COLORS["gray"], linestyle="--", linewidth=1)
    axes[1].set(ylabel="GPU memory (GiB)", ylim=(0, 84))
    axes[1].legend(frameon=False, ncol=2, loc="lower right")

    axes[2].plot(elapsed_h, one_min.gpu0_util_pct, color=COLORS["blue"], linewidth=1.1, label="GPU 0")
    axes[2].plot(elapsed_h, one_min.gpu1_util_pct, color=COLORS["orange"], linewidth=1.1, label="GPU 1")
    axes[2].set(xlabel="Hours since run start", ylabel="GPU utilization (%)", ylim=(0, 105))
    axes[2].legend(frameon=False, ncol=2, loc="lower right")

    for ax in axes:
        ax.grid(True, alpha=0.2)
    axes[0].set_title("Fast-WAM GRPO — resource history (1-minute maxima)")
    finish(fig, "resource-history.png")


if __name__ == "__main__":
    plot_success()
    plot_optimization()
    plot_resources()
