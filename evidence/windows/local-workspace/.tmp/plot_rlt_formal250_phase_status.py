from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PHASES = (
    (1, 135, "P1 collect", "#dbeafe"),
    (136, 154, "P2 ref + SAC", "#fef3c7"),
    (155, 190, "P3 student + ramp", "#dcfce7"),
    (191, 250, "P4 stable (projected)", "#f3e8ff"),
)
CURRENT_CYCLE = 162
P4_PROJECTED = 191


def add_phase_background(ax, *, label: bool = False) -> None:
    for start, end, name, color in PHASES:
        ax.axvspan(start - 0.5, end + 0.5, color=color, alpha=0.55, lw=0)
        if label:
            ax.text(
                (start + end) / 2,
                0.985,
                name,
                ha="center",
                va="top",
                transform=ax.get_xaxis_transform(),
                fontsize=8,
            )
    ax.axvline(136, color="#b45309", lw=1.2)
    ax.axvline(155, color="#15803d", lw=1.2)
    ax.axvline(P4_PROJECTED, color="#7e22ce", lw=1.2, ls=":")
    ax.axvline(CURRENT_CYCLE, color="#111827", lw=1.5, ls="--")


def scalar(ea: EventAccumulator, tag: str) -> list:
    return ea.Scalars(tag) if tag in ea.Tags()["scalars"] else []


def cycle_values(ea: EventAccumulator, tag: str) -> tuple[np.ndarray, np.ndarray]:
    events = scalar(ea, tag)
    return (
        np.asarray([event.step + 1 for event in events], dtype=float),
        np.asarray([event.value for event in events], dtype=float),
    )


def plot_success(ea: EventAccumulator, output: Path) -> None:
    cycles, success = cycle_values(ea, "env/success_once")
    eval_cycles, eval_success = cycle_values(ea, "eval/success_once")
    rates = success * 100
    rolling = pd.Series(rates).rolling(10, min_periods=1).mean().to_numpy()

    update_cycles, updates_run = cycle_values(ea, "train/rlt/critic_updates_run")
    update_after = np.cumsum(updates_run)
    switch_cycles, switch = cycle_values(ea, "train/replay/actor_switch_rate")

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1]},
        constrained_layout=True,
    )
    ax = axes[0]
    add_phase_background(ax, label=True)
    ax.scatter(
        cycles,
        rates,
        s=22,
        color="#60a5fa",
        edgecolor="#1d4ed8",
        alpha=0.45,
        label="train / cycle (8 episodes)",
        zorder=3,
    )
    ax.plot(cycles, rolling, color="#1d4ed8", lw=2.5, label="train rolling 10 cycles")
    ax.plot(
        eval_cycles,
        eval_success * 100,
        color="#dc2626",
        marker="D",
        ms=7,
        lw=1.8,
        label="deterministic student eval (20 episodes)",
        zorder=4,
    )
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(-2, 70)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    ax.set_title(
        "RLT Stage 2 success and four-phase schedule — complete cycle 162",
        fontsize=15,
        pad=20,
    )
    ax.text(
        3,
        64,
        "P1 156/1080 = 14.4%   |   P2 22/152 = 14.5%   |   "
        "P3 so far 3/64 = 4.69%   |   eval@150 1/20",
        fontsize=10,
        va="top",
    )

    ax2 = axes[1]
    add_phase_background(ax2)
    ax2.plot(
        update_cycles,
        update_after,
        color="#7c3aed",
        lw=2.2,
        label="cumulative critic updates",
    )
    ax2.axhline(30000, color="#15803d", lw=1.2, ls="--", label="student gate 30k")
    ax2.axhline(70000, color="#7e22ce", lw=1.2, ls=":", label="final weights 70k")
    switch_y = np.where(switch > 0.5, 7500, 0)
    ax2.fill_between(
        switch_cycles,
        0,
        switch_y,
        step="mid",
        color="#16a34a",
        alpha=0.25,
        label="student controls rollout",
    )
    ax2.set_ylabel("Critic updates")
    ax2.set_xlabel("Outer cycle")
    ax2.set_xlim(1, 250)
    ax2.set_ylim(0, 76000)
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(loc="upper left", ncol=4, fontsize=9)
    ax2.annotate(
        "current: 43.2k critic / 21.6k actor",
        xy=(162, 43200),
        xytext=(176, 49000),
        arrowprops={"arrowstyle": "->", "color": "#111827"},
        fontsize=9,
    )
    ax2.annotate(
        "P4 boundary projected near cycle 190–191",
        xy=(191, 70000),
        xytext=(198, 61000),
        arrowprops={"arrowstyle": "->", "color": "#7e22ce"},
        fontsize=9,
    )
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_optimization(ea: EventAccumulator, output: Path) -> None:
    tags = {
        "actor_loss": "train/sac/actor_loss",
        "critic_loss": "train/sac/critic_loss",
        "bc_loss": "train/actor/bc_loss",
        "weighted_bc": "train/actor/weighted_bc",
        "weighted_q": "train/actor/weighted_q",
        "actor_grad": "train/actor/grad_norm",
        "critic_grad": "train/critic/grad_norm",
        "q0": "train/actor/q_value_0",
        "q1": "train/actor/q_value_1",
        "q_data": "train/critic/q_data",
        "bc_weight": "train/actor/bc_weight",
        "q_weight": "train/actor/q_weight",
    }
    data = {name: cycle_values(ea, tag) for name, tag in tags.items()}
    fig, axes = plt.subplots(3, 2, figsize=(16, 15), sharex=True, constrained_layout=True)
    for ax in axes.flat:
        add_phase_background(ax)
        ax.set_xlim(134.5, 191.5)
        ax.grid(axis="y", alpha=0.25)

    ax = axes[0, 0]
    for key, label, color in (
        ("actor_loss", "actor loss", "#1d4ed8"),
        ("weighted_bc", "weighted BC", "#059669"),
        ("weighted_q", "weighted Q", "#dc2626"),
    ):
        x, y = data[key]
        ax.plot(x, y, lw=2, label=label, color=color)
    ax.set_title("Actor objective terms")
    ax.set_ylabel("Loss contribution")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    for key, label, color in (
        ("bc_loss", "BC loss", "#059669"),
        ("critic_loss", "critic loss", "#7c3aed"),
    ):
        x, y = data[key]
        ax.plot(x, y, lw=2, label=label, color=color)
    ax.set_yscale("log")
    ax.set_title("BC and critic losses (log scale)")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    for key, label, color in (
        ("actor_grad", "actor grad norm", "#ea580c"),
        ("critic_grad", "critic grad norm", "#2563eb"),
    ):
        x, y = data[key]
        ax.plot(x, y, lw=2, label=label, color=color)
    ax.axhline(10, color="#dc2626", lw=1.2, ls="--", label="clip 10")
    ax.set_yscale("log")
    ax.set_ylim(0.25, 12)
    ax.set_title("Gradient norms")
    ax.set_ylabel("Global norm")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    for key, label, color in (
        ("q0", "Q0(policy)", "#2563eb"),
        ("q1", "Q1(policy)", "#7c3aed"),
        ("q_data", "Q(data)", "#059669"),
    ):
        x, y = data[key]
        ax.plot(x, y, lw=2, label=label, color=color)
    ax.set_title("Twin-Q consistency")
    ax.set_ylabel("Q value")
    ax.legend(fontsize=9)

    ax = axes[2, 0]
    x_bc, y_bc = data["bc_weight"]
    x_q, y_q = data["q_weight"]
    ax.plot(x_bc, y_bc, lw=2, color="#059669", label="BC weight")
    ax.set_ylabel("BC weight")
    ax.set_title("Actor weight schedule")
    twin = ax.twinx()
    twin.plot(x_q, y_q, lw=2, color="#dc2626", label="Q weight")
    twin.set_ylabel("Q weight")
    lines = ax.get_lines() + twin.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], fontsize=9)

    ax = axes[2, 1]
    update_cycles, updates_run = cycle_values(ea, "train/rlt/critic_updates_run")
    pending_cycles, pending = cycle_values(ea, "train/rlt/pending_update_budget")
    ax.plot(update_cycles, updates_run, color="#7c3aed", lw=2, label="critic updates / cycle")
    ax.plot(pending_cycles, pending, color="#ea580c", lw=2, label="pending update debt")
    ax.set_title("Update cadence and debt")
    ax.set_ylabel("Updates")
    ax.legend(fontsize=9)
    for ax in axes[2, :]:
        ax.set_xlabel("Outer cycle")

    fig.suptitle(
        "RLT optimization metrics — phase boundaries, student switch, and current cycle",
        fontsize=16,
    )
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_resources(ea: EventAccumulator, resources: pd.DataFrame, output: Path) -> None:
    event_cycles, _ = cycle_values(ea, "env/success_once")
    event_times = np.asarray(
        [event.wall_time for event in scalar(ea, "env/success_once")],
        dtype=float,
    )
    sample_times = resources["unix_time"].to_numpy(dtype=float)
    cycle_axis = np.interp(sample_times, event_times, event_cycles)
    gib = 1024**3

    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True, constrained_layout=True)
    for ax in axes:
        add_phase_background(ax)
        ax.set_xlim(1, 191.5)
        ax.grid(axis="y", alpha=0.25)

    ax = axes[0]
    ax.plot(cycle_axis, resources["cgroup_current_bytes"] / gib, lw=1.3, label="cgroup current")
    ax.plot(cycle_axis, resources["cgroup_file_bytes"] / gib, lw=1.3, label="file cache")
    ax.plot(cycle_axis, resources["cgroup_anon_bytes"] / gib, lw=1.5, label="anonymous")
    ax.plot(cycle_axis, resources["matched_total_rss_kib"] / 1024**2, lw=1.5, label="matched RSS")
    ax.axhline(240, color="#dc2626", lw=1.2, ls="--", label="cgroup limit 240 GiB")
    ax.set_ylabel("Memory (GiB)")
    ax.set_title("Cgroup memory: cache is reclaiming while retained RSS remains elevated")
    ax.legend(loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=9)

    ax = axes[1]
    ax.plot(cycle_axis, resources["env_rss_kib"] / 1024**2, lw=1.5, label="EnvWorker RSS")
    ax.plot(cycle_axis, resources["rollout_rss_kib"] / 1024**2, lw=1.5, label="rollout RSS")
    ax.plot(cycle_axis, resources["matched_total_rss_kib"] / 1024**2, lw=1.5, label="matched total RSS")
    ax.set_ylabel("Process RSS (GiB)")
    ax.set_title("Retained process memory")
    ax.legend(loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=9)

    ax = axes[2]
    ax.plot(cycle_axis, resources["gpu0_used_mib"] / 1024, lw=1.2, label="GPU0 memory")
    ax.plot(cycle_axis, resources["gpu1_used_mib"] / 1024, lw=1.2, label="GPU1 memory")
    util0 = resources["gpu0_util_pct"].rolling(30, min_periods=1).mean()
    util1 = resources["gpu1_util_pct"].rolling(30, min_periods=1).mean()
    twin = ax.twinx()
    twin.plot(cycle_axis, util0, lw=1, alpha=0.65, label="GPU0 util, 60s mean")
    twin.plot(cycle_axis, util1, lw=1, alpha=0.65, label="GPU1 util, 60s mean")
    ax.set_ylabel("GPU memory (GiB)")
    twin.set_ylabel("GPU utilization (%)")
    ax.set_xlabel("Outer cycle (resource samples mapped from wall time)")
    ax.set_title("GPU memory and alternating utilization")
    lines = ax.get_lines() + twin.get_lines()
    ax.legend(
        lines,
        [line.get_label() for line in lines],
        loc="center left",
        bbox_to_anchor=(1.005, 0.5),
        fontsize=9,
    )
    fig.suptitle("RLT resource profile through complete cycle 162", fontsize=16)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def export_data(ea: EventAccumulator, resources: pd.DataFrame, output: Path) -> None:
    event_cycles, train_success = cycle_values(ea, "env/success_once")
    event_times = np.asarray(
        [event.wall_time for event in scalar(ea, "env/success_once")],
        dtype=float,
    )
    sample_times = resources["unix_time"].to_numpy(dtype=float)
    resource_cycles = np.interp(sample_times, event_times, event_cycles)
    sample_index = np.unique(
        np.linspace(0, len(resources) - 1, min(180, len(resources)), dtype=int)
    )

    metric_tags = {
        "actor_loss": "train/sac/actor_loss",
        "critic_loss": "train/sac/critic_loss",
        "bc_loss": "train/actor/bc_loss",
        "actor_grad": "train/actor/grad_norm",
        "critic_grad": "train/critic/grad_norm",
        "q0": "train/actor/q_value_0",
        "q1": "train/actor/q_value_1",
        "q_data": "train/critic/q_data",
        "bc_weight": "train/actor/bc_weight",
        "q_weight": "train/actor/q_weight",
    }
    metrics = {}
    for name, tag in metric_tags.items():
        cycles, values = cycle_values(ea, tag)
        metrics[name] = [
            [int(cycle), round(float(value), 7)]
            for cycle, value in zip(cycles, values, strict=True)
        ]

    update_cycles, updates_run = cycle_values(ea, "train/rlt/critic_updates_run")
    switch_cycles, switch = cycle_values(ea, "train/replay/actor_switch_rate")
    eval_cycles, eval_success = cycle_values(ea, "eval/success_once")
    payload = {
        "currentCycle": CURRENT_CYCLE,
        "projectedP4": P4_PROJECTED,
        "trainSuccess": [
            [int(cycle), int(round(value * 8))]
            for cycle, value in zip(event_cycles, train_success, strict=True)
        ],
        "evalSuccess": [
            [int(cycle), int(round(value * 20))]
            for cycle, value in zip(eval_cycles, eval_success, strict=True)
        ],
        "criticUpdates": [
            [int(cycle), int(total)]
            for cycle, total in zip(update_cycles, np.cumsum(updates_run), strict=True)
        ],
        "actorSwitch": [
            [int(cycle), int(value > 0.5)]
            for cycle, value in zip(switch_cycles, switch, strict=True)
        ],
        "metrics": metrics,
        "resources": [
            [
                round(float(resource_cycles[index]), 2),
                round(float(resources.iloc[index]["cgroup_current_bytes"] / 1024**3), 2),
                round(float(resources.iloc[index]["cgroup_anon_bytes"] / 1024**3), 2),
                round(float(resources.iloc[index]["cgroup_file_bytes"] / 1024**3), 2),
                round(float(resources.iloc[index]["matched_total_rss_kib"] / 1024**2), 2),
                round(float(resources.iloc[index]["gpu0_used_mib"] / 1024), 2),
                round(float(resources.iloc[index]["gpu1_used_mib"] / 1024), 2),
            ]
            for index in sample_index
        ],
    }
    output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    event_file = next(args.snapshot.glob("events.out.tfevents.*"))
    ea = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    ea.Reload()
    resources = pd.read_csv(args.snapshot / "resources.csv")

    plot_success(ea, args.output_dir / "rlt-success-phases-cycle162.png")
    plot_optimization(ea, args.output_dir / "rlt-optimization-cycle162.png")
    plot_resources(ea, resources, args.output_dir / "rlt-resources-cycle162.png")
    export_data(ea, resources, args.output_dir / "rlt-formal250-cycle162-data.json")


if __name__ == "__main__":
    main()
