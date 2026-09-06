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


TRAIN_EPISODES_PER_CYCLE = 8
EVAL_EPISODES = 20
FINAL_CYCLE = 480
RESUME_AFTER_CYCLE = 250


def load_accumulator(path: Path) -> EventAccumulator:
    event_files = list(path.glob("events.out.tfevents.*"))
    if len(event_files) != 1:
        raise RuntimeError(f"expected one event file under {path}, got {event_files}")
    accumulator = EventAccumulator(
        str(event_files[0]),
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()
    return accumulator


def merged_events(
    accumulators: list[EventAccumulator],
    tag: str,
) -> list[tuple[int, float, float]]:
    by_step: dict[int, tuple[int, float, float]] = {}
    for accumulator in accumulators:
        if tag not in accumulator.Tags().get("scalars", []):
            continue
        for point in accumulator.Scalars(tag):
            by_step[int(point.step)] = (
                int(point.step),
                float(point.value),
                float(point.wall_time),
            )
    return [by_step[step] for step in sorted(by_step)]


def cycle_values(
    accumulators: list[EventAccumulator],
    tag: str,
) -> tuple[np.ndarray, np.ndarray]:
    points = merged_events(accumulators, tag)
    return (
        np.asarray([step + 1 for step, _, _ in points], dtype=int),
        np.asarray([value for _, value, _ in points], dtype=float),
    )


def first_cycle(
    accumulators: list[EventAccumulator],
    tag: str,
    predicate,
) -> int:
    cycles, values = cycle_values(accumulators, tag)
    matches = cycles[np.asarray([predicate(value) for value in values])]
    if not len(matches):
        raise RuntimeError(f"no phase boundary found for {tag}")
    return int(matches[0])


def phase_boundaries(accumulators: list[EventAccumulator]) -> dict[str, int]:
    return {
        "p2": first_cycle(
            accumulators,
            "train/rlt/critic_updates_run",
            lambda value: value > 0,
        ),
        "p3": first_cycle(
            accumulators,
            "train/replay/actor_switch_rate",
            lambda value: value >= 0.5,
        ),
        "p4": first_cycle(
            accumulators,
            "train/actor/actor_weight_ramp_progress",
            lambda value: value >= 1.0 - 1e-6,
        ),
    }


def phase_spans(boundaries: dict[str, int]) -> list[tuple[int, int, str, str]]:
    return [
        (1, boundaries["p2"] - 1, "P1 reference collect", "#dbeafe"),
        (
            boundaries["p2"],
            boundaries["p3"] - 1,
            "P2 SAC (reference)",
            "#fef3c7",
        ),
        (
            boundaries["p3"],
            boundaries["p4"] - 1,
            "P3 student ramp",
            "#dcfce7",
        ),
        (boundaries["p4"], FINAL_CYCLE, "P4 stable student", "#f3e8ff"),
    ]


def shade_phases(
    ax,
    boundaries: dict[str, int],
    *,
    phase_labels: bool = False,
) -> None:
    for start, end, label, color in phase_spans(boundaries):
        ax.axvspan(start - 0.5, end + 0.5, color=color, alpha=0.45, lw=0)
        if phase_labels and end - start >= 12:
            label_y = 0.965 if start == boundaries["p3"] else 0.985
            ax.text(
                (start + end) / 2,
                label_y,
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
        ax.axvline(cycle - 0.5, color=color, lw=1.05)
    ax.axvline(
        RESUME_AFTER_CYCLE + 0.5,
        color="#111827",
        lw=1.3,
        ls="--",
        label="process resume after cycle 250",
    )


def aggregate_success(
    cycles: np.ndarray,
    values: np.ndarray,
    start: int,
    end: int,
) -> dict[str, float | int]:
    mask = (cycles >= start) & (cycles <= end)
    successes = int(
        np.rint(values[mask] * TRAIN_EPISODES_PER_CYCLE).sum()
    )
    episodes = int(mask.sum() * TRAIN_EPISODES_PER_CYCLE)
    return {
        "start": start,
        "end": end,
        "successes": successes,
        "episodes": episodes,
        "rate": successes / episodes if episodes else float("nan"),
    }


def plot_success(
    accumulators: list[EventAccumulator],
    boundaries: dict[str, int],
    output: Path,
) -> dict:
    cycles, success = cycle_values(accumulators, "env/success_once")
    eval_cycles, eval_success = cycle_values(accumulators, "eval/success_once")
    update_cycles, update_step = cycle_values(
        accumulators,
        "train/rlt/update_step",
    )
    update_run_cycles, updates_run = cycle_values(
        accumulators,
        "train/rlt/critic_updates_run",
    )
    if not np.array_equal(update_cycles, update_run_cycles):
        raise RuntimeError("update_step and critic_updates_run cycle grids differ")
    post_update_step = update_step + updates_run
    rolling10 = (
        pd.Series(success * 100)
        .rolling(10, min_periods=1)
        .mean()
        .to_numpy()
    )
    rolling50 = (
        pd.Series(success * 100)
        .rolling(50, min_periods=1)
        .mean()
        .to_numpy()
    )

    phase_rows = []
    for start, end, label, _ in phase_spans(boundaries):
        row = aggregate_success(cycles, success, start, end)
        row["phase"] = label.split()[0]
        phase_rows.append(row)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
        constrained_layout=True,
    )
    ax = axes[0]
    shade_phases(ax, boundaries, phase_labels=True)
    ax.scatter(
        cycles,
        success * 100,
        s=13,
        color="#60a5fa",
        edgecolor="#1d4ed8",
        linewidth=0.35,
        alpha=0.32,
        label="train cycle (8 episodes)",
        zorder=3,
    )
    ax.plot(
        cycles,
        rolling10,
        color="#1d4ed8",
        lw=2.0,
        label="train rolling 10 cycles",
    )
    ax.plot(
        cycles,
        rolling50,
        color="#0f766e",
        lw=1.7,
        label="train rolling 50 cycles",
    )
    ax.plot(
        eval_cycles,
        eval_success * 100,
        color="#dc2626",
        marker="D",
        ms=5,
        lw=1.4,
        label="deterministic eval (20 fixed seeds)",
        zorder=4,
    )
    for index, (cycle, value) in enumerate(
        zip(eval_cycles, eval_success, strict=True)
    ):
        offset = 8 if index % 2 == 0 else -15
        horizontal_offset = -18 if cycle == FINAL_CYCLE else 0
        ax.annotate(
            f"{int(round(value * EVAL_EPISODES))}/20",
            (cycle, value * 100),
            textcoords="offset points",
            xytext=(horizontal_offset, offset),
            ha="right" if cycle == FINAL_CYCLE else "center",
            fontsize=7.5,
            color="#991b1b",
        )
    phase_text = " | ".join(
        f"{row['phase']} {row['successes']}/{row['episodes']}="
        f"{100 * float(row['rate']):.1f}%"
        for row in phase_rows
    )
    ax.text(
        0.01,
        0.925,
        phase_text,
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
    )
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(-7, 106)
    ax.set_xlim(1, FINAL_CYCLE)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.set_title(
        "RLT Stage 2 success through complete cycle 480",
        fontsize=14,
    )

    ax = axes[1]
    shade_phases(ax, boundaries)
    ax.plot(
        update_cycles,
        post_update_step,
        color="#7c3aed",
        lw=2.2,
        label="lifetime critic update step",
    )
    ax.axhline(
        30_000,
        color="#15803d",
        lw=1.1,
        ls="--",
        label="student gate (30k)",
    )
    ax.axhline(
        70_000,
        color="#7e22ce",
        lw=1.1,
        ls=":",
        label="weight ramp complete (70k)",
    )
    ax.annotate(
        f"{int(post_update_step[-1]):,}",
        (update_cycles[-1], post_update_step[-1]),
        textcoords="offset points",
        xytext=(-42, 7),
        fontsize=8,
    )
    ax.set_ylabel("Critic update step")
    ax.set_xlabel("Outer cycle")
    ax.set_xlim(1, FINAL_CYCLE)
    ax.set_ylim(0, post_update_step[-1] * 1.08)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper left", fontsize=8, ncol=4)
    fig.savefig(output, dpi=170)
    plt.close(fig)

    return {
        "cycles": int(len(cycles)),
        "latest_rolling10": float(rolling10[-1] / 100),
        "latest_rolling50": float(rolling50[-1] / 100),
        "phase_rows": phase_rows,
        "eval": [
            {
                "cycle": int(cycle),
                "successes": int(round(value * EVAL_EPISODES)),
                "episodes": EVAL_EPISODES,
            }
            for cycle, value in zip(eval_cycles, eval_success, strict=True)
        ],
        "final_update_step": int(round(post_update_step[-1])),
    }


def plot_optimization(
    accumulators: list[EventAccumulator],
    boundaries: dict[str, int],
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
    }
    data = {
        name: cycle_values(accumulators, tag)
        for name, tag in tags.items()
    }
    start_cycle = boundaries["p2"]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 9),
        sharex=True,
        constrained_layout=True,
    )
    for ax in axes.flat:
        shade_phases(ax, boundaries)
        ax.set_xlim(start_cycle - 1, FINAL_CYCLE + 1)
        ax.grid(axis="y", alpha=0.22)

    ax = axes[0, 0]
    for key, label, color in (
        ("actor_loss", "actor loss", "#1d4ed8"),
        ("weighted_bc", "weighted BC", "#059669"),
        ("weighted_q", "weighted Q", "#dc2626"),
    ):
        x, y = data[key]
        ax.plot(x, y, lw=1.8, label=label, color=color)
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
        ax.plot(x, y, lw=1.8, label=label, color=color)
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
        ax.plot(x, y, lw=1.8, label=label, color=color)
    ax.axhline(10, color="#dc2626", lw=1.1, ls="--", label="clip threshold 10")
    ax.set_yscale("log")
    ax.set_ylim(0.08, 12)
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
        ax.plot(x, y, lw=1.8, label=label, color=color)
    ax.set_title("Twin-Q values")
    ax.set_ylabel("Q value")
    ax.set_xlabel("Outer cycle")
    ax.legend(fontsize=8)
    fig.suptitle(
        "RLT optimization metrics through complete cycle 480",
        fontsize=14,
    )
    fig.savefig(output, dpi=170)
    plt.close(fig)

    summary: dict[str, dict[str, float | int]] = {}
    for name, (cycles, values) in data.items():
        if not len(values):
            continue
        summary[name] = {
            "first_cycle": int(cycles[0]),
            "last_cycle": int(cycles[-1]),
            "first": float(values[0]),
            "last": float(values[-1]),
            "min": float(values.min()),
            "max": float(values.max()),
            "last10_mean": float(values[-10:].mean()),
            "last20_mean": float(values[-20:].mean()),
        }
    return summary


def map_resource_cycles(
    resources: pd.DataFrame,
    accumulator: EventAccumulator,
) -> pd.DataFrame:
    points = merged_events([accumulator], "env/success_once")
    event_cycles = np.asarray([step + 1 for step, _, _ in points], dtype=float)
    event_times = np.asarray([wall_time for _, _, wall_time in points], dtype=float)
    mapped = resources.copy()
    mapped["outer_cycle"] = np.interp(
        mapped["unix_time"].to_numpy(dtype=float),
        event_times,
        event_cycles,
    )
    return mapped


def plot_resources(
    first: pd.DataFrame,
    resumed: pd.DataFrame,
    first_accumulator: EventAccumulator,
    resumed_accumulator: EventAccumulator,
    boundaries: dict[str, int],
    output: Path,
) -> dict:
    frames = [
        map_resource_cycles(first, first_accumulator).assign(run="cycles 1-250"),
        map_resource_cycles(resumed, resumed_accumulator).assign(run="cycles 251-480"),
    ]
    resources = pd.concat(frames, ignore_index=True)
    gib = 1024**3

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15, 11),
        sharex=True,
        constrained_layout=True,
    )
    for ax in axes:
        shade_phases(ax, boundaries)
        ax.set_xlim(1, FINAL_CYCLE)
        ax.grid(axis="y", alpha=0.22)

    ax = axes[0]
    for column, divisor, label, color in (
        ("cgroup_current_bytes", gib, "cgroup current", "#2563eb"),
        ("cgroup_file_bytes", gib, "file cache", "#ea580c"),
        ("cgroup_anon_bytes", gib, "anonymous memory", "#059669"),
        ("matched_total_rss_kib", 1024**2, "matched process RSS", "#dc2626"),
    ):
        ax.plot(
            resources["outer_cycle"],
            resources[column] / divisor,
            lw=1.15,
            label=label,
            color=color,
        )
    ax.axhline(
        240,
        color="#dc2626",
        lw=1.1,
        ls="--",
        label="cgroup limit 240 GiB",
    )
    ax.set_ylabel("GiB")
    ax.set_title("Cgroup memory and retained process RSS")
    ax.legend(loc="upper left", ncol=3, fontsize=8)

    ax = axes[1]
    for column, label, color in (
        ("env_rss_kib", "EnvWorker RSS", "#2563eb"),
        ("rollout_rss_kib", "rollout RSS", "#ea580c"),
        ("actor_rss_kib", "actor RSS", "#7c3aed"),
        ("matched_total_rss_kib", "matched total RSS", "#059669"),
    ):
        ax.plot(
            resources["outer_cycle"],
            resources[column] / 1024**2,
            lw=1.15,
            label=label,
            color=color,
        )
    ax.set_ylabel("GiB")
    ax.set_title("Matched training-process RSS")
    ax.legend(loc="upper left", ncol=4, fontsize=8)

    ax = axes[2]
    ax.plot(
        resources["outer_cycle"],
        resources["gpu0_used_mib"] / 1024,
        lw=1.0,
        label="GPU0 memory",
        color="#2563eb",
    )
    ax.plot(
        resources["outer_cycle"],
        resources["gpu1_used_mib"] / 1024,
        lw=1.0,
        label="GPU1 memory",
        color="#ea580c",
    )
    twin = ax.twinx()
    for column, label, color in (
        ("gpu0_util_pct", "GPU0 util (60-s mean)", "#7c3aed"),
        ("gpu1_util_pct", "GPU1 util (60-s mean)", "#059669"),
    ):
        rolling = (
            resources.groupby("run", sort=False)[column]
            .rolling(30, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        twin.plot(
            resources["outer_cycle"],
            rolling,
            lw=1.0,
            alpha=0.78,
            label=label,
            color=color,
        )
    ax.set_ylabel("GPU memory (GiB)")
    twin.set_ylabel("GPU utilization (%)")
    ax.set_xlabel("Outer cycle (resource time mapped to training metrics)")
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
        "RLT resource profile across the original and resumed processes",
        fontsize=14,
    )
    fig.savefig(output, dpi=170)
    plt.close(fig)

    def summary_for(frame: pd.DataFrame) -> dict:
        active = frame[
            (frame["gpu0_used_mib"] > 0) | (frame["gpu1_used_mib"] > 0)
        ]
        return {
            "rows": int(len(frame)),
            "duration_seconds": int(
                frame["unix_time"].iloc[-1] - frame["unix_time"].iloc[0]
            ),
            "cgroup_current_peak_gib": float(
                frame["cgroup_current_bytes"].max() / gib
            ),
            "cgroup_anon_peak_gib": float(
                frame["cgroup_anon_bytes"].max() / gib
            ),
            "cgroup_file_peak_gib": float(
                frame["cgroup_file_bytes"].max() / gib
            ),
            "matched_rss_peak_gib": float(
                frame["matched_total_rss_kib"].max() / 1024**2
            ),
            "env_rss_peak_gib": float(
                frame["env_rss_kib"].max() / 1024**2
            ),
            "gpu0_memory_peak_gib": float(
                frame["gpu0_used_mib"].max() / 1024
            ),
            "gpu1_memory_peak_gib": float(
                frame["gpu1_used_mib"].max() / 1024
            ),
            "gpu0_active_util_mean": float(active["gpu0_util_pct"].mean()),
            "gpu1_active_util_mean": float(active["gpu1_util_pct"].mean()),
            "high_event_delta": int(
                frame["cgroup_high_events"].iloc[-1]
                - frame["cgroup_high_events"].iloc[0]
            ),
            "max_event_delta": int(
                frame["cgroup_max_events"].iloc[-1]
                - frame["cgroup_max_events"].iloc[0]
            ),
            "oom_delta": int(
                frame["cgroup_oom_events"].iloc[-1]
                - frame["cgroup_oom_events"].iloc[0]
            ),
            "oom_kill_delta": int(
                frame["cgroup_oom_kill_events"].iloc[-1]
                - frame["cgroup_oom_kill_events"].iloc[0]
            ),
        }

    return {
        "cycles_1_250": summary_for(first),
        "cycles_251_480": summary_for(resumed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("original_snapshot", type=Path)
    parser.add_argument("resume_snapshot", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    original_event_dir = args.original_snapshot / "runtime" / "tensorboard"
    original_resources = pd.read_csv(
        args.original_snapshot / "runtime" / "resources.csv"
    )
    resume_resources = pd.read_csv(args.resume_snapshot / "resources.csv")
    original_accumulator = load_accumulator(original_event_dir)
    resume_accumulator = load_accumulator(args.resume_snapshot)
    accumulators = [original_accumulator, resume_accumulator]
    boundaries = phase_boundaries(accumulators)

    payload = {
        "phase_boundaries": boundaries,
        "success": plot_success(
            accumulators,
            boundaries,
            args.output_dir / "rlt-success-complete480.png",
        ),
        "optimization": plot_optimization(
            accumulators,
            boundaries,
            args.output_dir / "rlt-optimization-complete480.png",
        ),
        "resources": plot_resources(
            original_resources,
            resume_resources,
            original_accumulator,
            resume_accumulator,
            boundaries,
            args.output_dir / "rlt-resources-complete480.png",
        ),
    }
    summary_path = args.output_dir / "rlt-summary-complete480.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(summary_path), **payload["success"]}))


if __name__ == "__main__":
    main()
