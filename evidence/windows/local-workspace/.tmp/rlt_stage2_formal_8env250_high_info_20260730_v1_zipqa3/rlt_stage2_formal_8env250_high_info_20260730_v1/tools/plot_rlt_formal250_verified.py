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


EPISODES_PER_TRAIN_CYCLE = 8
EPISODES_PER_EVAL = 20
PLANNED_CYCLES = 250


def events(ea: EventAccumulator, tag: str) -> list:
    if tag not in ea.Tags()["scalars"]:
        return []
    return ea.Scalars(tag)


def cycle_values(ea: EventAccumulator, tag: str) -> tuple[np.ndarray, np.ndarray]:
    items = events(ea, tag)
    return (
        np.asarray([item.step + 1 for item in items], dtype=int),
        np.asarray([item.value for item in items], dtype=float),
    )


def first_cycle_where(
    ea: EventAccumulator, tag: str, predicate
) -> int | None:
    cycles, values = cycle_values(ea, tag)
    matches = cycles[np.asarray([predicate(value) for value in values])]
    return int(matches[0]) if len(matches) else None


def load(snapshot: Path) -> tuple[EventAccumulator, pd.DataFrame]:
    event_files = list(snapshot.glob("events.out.tfevents.*"))
    if len(event_files) != 1:
        raise RuntimeError(f"expected exactly one event file, got {event_files}")
    ea = EventAccumulator(str(event_files[0]), size_guidance={"scalars": 0})
    ea.Reload()
    resources = pd.read_csv(snapshot / "resources.csv")
    return ea, resources


def phase_contract(ea: EventAccumulator) -> dict[str, int]:
    optimize_start = first_cycle_where(
        ea, "train/rlt/critic_updates_run", lambda value: value > 0
    )
    student_start = first_cycle_where(
        ea, "train/replay/actor_switch_rate", lambda value: value >= 0.5
    )
    stable_start = first_cycle_where(
        ea,
        "train/actor/actor_weight_ramp_progress",
        lambda value: value >= 1.0 - 1e-6,
    )
    if optimize_start is None or student_start is None:
        raise RuntimeError("optimization or student phase boundary is absent")
    return {
        "p2": optimize_start,
        "p3": student_start,
        "p4": stable_start or PLANNED_CYCLES + 1,
    }


def phase_spans(boundaries: dict[str, int]) -> list[tuple[int, int, str, str]]:
    return [
        (1, boundaries["p2"] - 1, "P1 reference collect", "#dbeafe"),
        (
            boundaries["p2"],
            boundaries["p3"] - 1,
            "P2 reference + SAC",
            "#fef3c7",
        ),
        (
            boundaries["p3"],
            boundaries["p4"] - 1,
            "P3 student + ramp",
            "#dcfce7",
        ),
        (
            boundaries["p4"],
            PLANNED_CYCLES,
            "P4 stable student",
            "#f3e8ff",
        ),
    ]


def shade_phases(
    ax,
    boundaries: dict[str, int],
    *,
    labels: bool = False,
    x_max: int = PLANNED_CYCLES,
) -> None:
    for start, end, label, color in phase_spans(boundaries):
        if start > x_max:
            continue
        end = min(end, x_max)
        ax.axvspan(start - 0.5, end + 0.5, color=color, alpha=0.48, lw=0)
        if labels and end - start >= 5:
            ax.text(
                (start + end) / 2,
                0.985,
                label,
                ha="center",
                va="top",
                transform=ax.get_xaxis_transform(),
                fontsize=8,
            )
    for cycle, color in (
        (boundaries["p2"], "#b45309"),
        (boundaries["p3"], "#15803d"),
        (boundaries["p4"], "#7e22ce"),
    ):
        if cycle <= x_max:
            ax.axvline(cycle - 0.5, color=color, lw=1.1)


def aggregate_success(
    cycles: np.ndarray,
    values: np.ndarray,
    start: int,
    end: int,
) -> tuple[int, int, float]:
    mask = (cycles >= start) & (cycles <= end)
    successes = int(np.rint(values[mask] * EPISODES_PER_TRAIN_CYCLE).sum())
    episodes = int(mask.sum() * EPISODES_PER_TRAIN_CYCLE)
    rate = successes / episodes if episodes else float("nan")
    return successes, episodes, rate


def plot_success(
    ea: EventAccumulator,
    boundaries: dict[str, int],
    output: Path,
) -> dict:
    cycles, success = cycle_values(ea, "env/success_once")
    eval_cycles, eval_success = cycle_values(ea, "eval/success_once")
    current = int(cycles[-1])
    rolling = (
        pd.Series(success * 100)
        .rolling(10, min_periods=1)
        .mean()
        .to_numpy()
    )
    update_cycles, updates_run = cycle_values(ea, "train/rlt/critic_updates_run")
    cumulative_updates = np.cumsum(updates_run)

    phase_rows = []
    for start, end, label, _ in phase_spans(boundaries):
        observed_end = min(end, current)
        if observed_end < start:
            continue
        successes, episodes, rate = aggregate_success(
            cycles, success, start, observed_end
        )
        phase_rows.append(
            {
                "phase": label.split()[0],
                "start": start,
                "end": observed_end,
                "successes": successes,
                "episodes": episodes,
                "rate": rate,
            }
        )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1]},
        constrained_layout=True,
    )
    ax = axes[0]
    shade_phases(ax, boundaries, labels=True)
    ax.scatter(
        cycles,
        success * 100,
        s=18,
        color="#60a5fa",
        edgecolor="#1d4ed8",
        alpha=0.42,
        label="train cycle (8 episodes)",
        zorder=3,
    )
    ax.plot(
        cycles,
        rolling,
        color="#1d4ed8",
        lw=2.4,
        label="train rolling 10 cycles",
    )
    ax.plot(
        eval_cycles,
        eval_success * 100,
        color="#dc2626",
        marker="D",
        ms=6,
        lw=1.6,
        label="deterministic eval (20 episodes)",
        zorder=4,
    )
    for cycle, value in zip(eval_cycles, eval_success, strict=True):
        ax.annotate(
            f"{int(round(value * EPISODES_PER_EVAL))}/20",
            (cycle, value * 100),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
            color="#991b1b",
        )
    phase_text = " | ".join(
        f"{row['phase']} {row['successes']}/{row['episodes']}="
        f"{100 * row['rate']:.1f}%"
        for row in phase_rows
    )
    ax.text(
        0.01,
        0.92,
        phase_text,
        transform=ax.transAxes,
        fontsize=9,
        va="top",
    )
    ax.axvline(current, color="#111827", lw=1.3, ls="--")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(-3, 105)
    ax.set_xlim(1, PLANNED_CYCLES)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="lower right", ncol=1, fontsize=8)
    ax.set_title(f"RLT Stage 2 success through complete cycle {current}", fontsize=14)

    ax = axes[1]
    shade_phases(ax, boundaries)
    ax.plot(
        update_cycles,
        cumulative_updates,
        color="#7c3aed",
        lw=2.2,
        label="cumulative critic updates",
    )
    ax.axhline(30000, color="#15803d", lw=1.1, ls="--", label="student gate 30k")
    ax.axhline(70000, color="#7e22ce", lw=1.1, ls=":", label="final weights 70k")
    ax.axvline(current, color="#111827", lw=1.3, ls="--")
    ax.annotate(
        f"{int(cumulative_updates[-1]):,}",
        (update_cycles[-1], cumulative_updates[-1]),
        textcoords="offset points",
        xytext=(6, 3),
        fontsize=8,
    )
    ax.set_ylabel("Critic updates")
    ax.set_xlabel("Outer cycle")
    ax.set_xlim(1, PLANNED_CYCLES)
    ax.set_ylim(0, max(76000, cumulative_updates[-1] * 1.08))
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper left", ncol=3, fontsize=8)

    fig.savefig(output, dpi=160)
    plt.close(fig)
    return {
        "current_cycle": current,
        "phase_rows": phase_rows,
        "latest_rolling10": float(rolling[-1] / 100),
        "eval": [
            {
                "cycle": int(cycle),
                "successes": int(round(value * EPISODES_PER_EVAL)),
                "episodes": EPISODES_PER_EVAL,
            }
            for cycle, value in zip(eval_cycles, eval_success, strict=True)
        ],
        "critic_updates": int(cumulative_updates[-1]),
    }


def plot_optimization(
    ea: EventAccumulator,
    boundaries: dict[str, int],
    current: int,
    output: Path,
) -> dict:
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
        "ramp": "train/actor/actor_weight_ramp_progress",
        "pending": "train/rlt/pending_update_budget",
    }
    data = {name: cycle_values(ea, tag) for name, tag in tags.items()}
    x_start = boundaries["p2"] - 1

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True, constrained_layout=True)
    for ax in axes.flat:
        shade_phases(ax, boundaries, x_max=current)
        ax.axvline(current, color="#111827", lw=1.2, ls="--")
        ax.set_xlim(x_start, current + 1)
        ax.grid(axis="y", alpha=0.22)

    ax = axes[0, 0]
    for key, label, color in (
        ("actor_loss", "actor loss", "#1d4ed8"),
        ("weighted_bc", "weighted BC", "#059669"),
        ("weighted_q", "weighted Q", "#dc2626"),
    ):
        x, y = data[key]
        ax.plot(x, y, lw=2, label=label, color=color)
    ax.axhline(0, color="#6b7280", lw=0.8)
    ax.set_title("Actor objective terms")
    ax.set_ylabel("Contribution")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for key, label, color in (
        ("bc_loss", "BC loss", "#059669"),
        ("critic_loss", "critic loss", "#7c3aed"),
    ):
        x, y = data[key]
        ax.plot(x, y, lw=2, label=label, color=color)
    ax.set_yscale("log")
    ax.set_title("BC and critic losses")
    ax.set_ylabel("Loss (log scale)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for key, label, color in (
        ("actor_grad", "actor grad norm", "#ea580c"),
        ("critic_grad", "critic grad norm", "#2563eb"),
    ):
        x, y = data[key]
        ax.plot(x, y, lw=2, label=label, color=color)
    ax.axhline(10, color="#dc2626", lw=1.1, ls="--", label="clip 10")
    ax.set_yscale("log")
    ax.set_ylim(0.12, 12)
    ax.set_title("Gradient norms")
    ax.set_ylabel("Global norm (log scale)")
    ax.set_xlabel("Outer cycle")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for key, label, color in (
        ("q0", "Q0(policy)", "#2563eb"),
        ("q1", "Q1(policy)", "#7c3aed"),
        ("q_data", "Q(data)", "#059669"),
    ):
        x, y = data[key]
        ax.plot(x, y, lw=2, label=label, color=color)
    ax.set_title("Twin-Q values")
    ax.set_ylabel("Q value")
    ax.set_xlabel("Outer cycle")
    ax.legend(fontsize=8)

    fig.suptitle(
        f"RLT optimization metrics through complete cycle {current}",
        fontsize=14,
    )
    fig.savefig(output, dpi=160)
    plt.close(fig)

    latest = {}
    for name, (cycles, values) in data.items():
        if len(values):
            latest[name] = {
                "cycle": int(cycles[-1]),
                "value": float(values[-1]),
            }
    return latest


def plot_resources(
    ea: EventAccumulator,
    resources: pd.DataFrame,
    boundaries: dict[str, int],
    current: int,
    output: Path,
) -> dict:
    success_events = events(ea, "env/success_once")
    event_cycles = np.asarray([event.step + 1 for event in success_events], dtype=float)
    event_times = np.asarray([event.wall_time for event in success_events], dtype=float)
    sample_times = resources["unix_time"].to_numpy(dtype=float)
    cycle_axis = np.interp(sample_times, event_times, event_cycles)
    gib = 1024**3

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    for ax in axes:
        shade_phases(ax, boundaries, x_max=current)
        ax.axvline(current, color="#111827", lw=1.2, ls="--")
        ax.set_xlim(1, current + 1)
        ax.grid(axis="y", alpha=0.22)

    ax = axes[0]
    series = (
        ("cgroup_current_bytes", gib, "cgroup current", "#2563eb"),
        ("cgroup_file_bytes", gib, "file cache", "#ea580c"),
        ("cgroup_anon_bytes", gib, "anonymous", "#059669"),
        ("matched_total_rss_kib", 1024**2, "matched process RSS", "#dc2626"),
    )
    for column, divisor, label, color in series:
        ax.plot(cycle_axis, resources[column] / divisor, lw=1.3, label=label, color=color)
    ax.axhline(240, color="#dc2626", lw=1.1, ls="--", label="cgroup limit 240 GiB")
    ax.set_ylabel("GiB")
    ax.set_title("Cgroup memory and retained process RSS")
    ax.legend(loc="upper left", ncol=3, fontsize=8)

    ax = axes[1]
    ax.plot(
        cycle_axis,
        resources["env_rss_kib"] / 1024**2,
        lw=1.5,
        label="EnvWorker RSS",
        color="#2563eb",
    )
    ax.plot(
        cycle_axis,
        resources["rollout_rss_kib"] / 1024**2,
        lw=1.5,
        label="rollout RSS",
        color="#ea580c",
    )
    ax.plot(
        cycle_axis,
        resources["matched_total_rss_kib"] / 1024**2,
        lw=1.5,
        label="matched total RSS",
        color="#059669",
    )
    ax.set_ylabel("GiB")
    ax.set_title("Matched process RSS")
    ax.legend(loc="upper left", ncol=3, fontsize=8)

    ax = axes[2]
    ax.plot(
        cycle_axis,
        resources["gpu0_used_mib"] / 1024,
        lw=1.2,
        label="GPU0 memory",
        color="#2563eb",
    )
    ax.plot(
        cycle_axis,
        resources["gpu1_used_mib"] / 1024,
        lw=1.2,
        label="GPU1 memory",
        color="#ea580c",
    )
    util0 = resources["gpu0_util_pct"].rolling(30, min_periods=1).mean()
    util1 = resources["gpu1_util_pct"].rolling(30, min_periods=1).mean()
    twin = ax.twinx()
    twin.plot(cycle_axis, util0, lw=1, alpha=0.75, label="GPU0 util (60s mean)", color="#7c3aed")
    twin.plot(cycle_axis, util1, lw=1, alpha=0.75, label="GPU1 util (60s mean)", color="#059669")
    ax.set_ylabel("GPU memory (GiB)")
    twin.set_ylabel("GPU utilization (%)")
    ax.set_xlabel("Outer cycle (resource samples mapped from wall time)")
    ax.set_title("GPU memory and utilization")
    lines = [
        line
        for line in ax.get_lines() + twin.get_lines()
        if not line.get_label().startswith("_")
    ]
    ax.legend(
        lines,
        [line.get_label() for line in lines],
        loc="upper left",
        ncol=2,
        fontsize=8,
    )

    fig.suptitle(
        f"RLT resource profile through complete cycle {current}",
        fontsize=14,
    )
    fig.savefig(output, dpi=160)
    plt.close(fig)

    def latest_and_peak(column: str, divisor: float) -> dict[str, float]:
        values = resources[column].to_numpy(dtype=float) / divisor
        return {"latest": float(values[-1]), "peak": float(values.max())}

    return {
        "snapshot_unix_time": float(resources["unix_time"].iloc[-1]),
        "cgroup_current_gib": latest_and_peak("cgroup_current_bytes", gib),
        "cgroup_anon_gib": latest_and_peak("cgroup_anon_bytes", gib),
        "cgroup_file_gib": latest_and_peak("cgroup_file_bytes", gib),
        "matched_rss_gib": latest_and_peak("matched_total_rss_kib", 1024**2),
        "env_rss_gib": latest_and_peak("env_rss_kib", 1024**2),
        "gpu0_used_gib": latest_and_peak("gpu0_used_mib", 1024),
        "gpu1_used_gib": latest_and_peak("gpu1_used_mib", 1024),
        "gpu0_mean_util": float(resources["gpu0_util_pct"].mean()),
        "gpu1_mean_util": float(resources["gpu1_util_pct"].mean()),
        "memory_max_event_delta": int(
            resources["cgroup_max_events"].iloc[-1]
            - resources["cgroup_max_events"].iloc[0]
        ),
        "oom_delta": int(
            resources["cgroup_oom_events"].iloc[-1]
            - resources["cgroup_oom_events"].iloc[0]
        ),
        "oom_kill_delta": int(
            resources["cgroup_oom_kill_events"].iloc[-1]
            - resources["cgroup_oom_kill_events"].iloc[0]
        ),
    }


def export_inline_data(
    ea: EventAccumulator,
    boundaries: dict[str, int],
    summary: dict,
    output: Path,
) -> None:
    cycles, success = cycle_values(ea, "env/success_once")
    eval_cycles, eval_success = cycle_values(ea, "eval/success_once")
    update_cycles, updates_run = cycle_values(ea, "train/rlt/critic_updates_run")
    payload = {
        "currentCycle": int(cycles[-1]),
        "plannedCycles": PLANNED_CYCLES,
        "boundaries": boundaries,
        "train": [
            [int(cycle), int(round(value * EPISODES_PER_TRAIN_CYCLE))]
            for cycle, value in zip(cycles, success, strict=True)
        ],
        "eval": [
            [int(cycle), int(round(value * EPISODES_PER_EVAL))]
            for cycle, value in zip(eval_cycles, eval_success, strict=True)
        ],
        "criticUpdates": [
            [int(cycle), int(total)]
            for cycle, total in zip(update_cycles, np.cumsum(updates_run), strict=True)
        ],
        "summary": summary,
    }
    output.write_text(
        json.dumps(payload, separators=(",", ":")),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ea, resources = load(args.snapshot)
    boundaries = phase_contract(ea)
    success_summary = plot_success(
        ea,
        boundaries,
        args.output_dir / f"rlt-success-{args.label}.png",
    )
    current = int(success_summary["current_cycle"])
    optimization_summary = plot_optimization(
        ea,
        boundaries,
        current,
        args.output_dir / f"rlt-optimization-{args.label}.png",
    )
    resource_summary = plot_resources(
        ea,
        resources,
        boundaries,
        current,
        args.output_dir / f"rlt-resources-{args.label}.png",
    )
    summary = {
        "boundaries": boundaries,
        "success": success_summary,
        "optimization": optimization_summary,
        "resources": resource_summary,
    }
    summary_path = args.output_dir / f"rlt-summary-{args.label}.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    export_inline_data(
        ea,
        boundaries,
        summary,
        args.output_dir / f"rlt-inline-data-{args.label}.json",
    )
    print(
        json.dumps(
            {
                "current_cycle": current,
                "boundaries": boundaries,
                "summary": str(summary_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
