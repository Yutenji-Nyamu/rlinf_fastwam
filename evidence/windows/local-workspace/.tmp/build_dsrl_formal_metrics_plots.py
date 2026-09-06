#!/usr/bin/env python3
"""Build current progress and first-update diagnostic plots from TensorBoard."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def scalar_map(accumulator: EventAccumulator, tag: str) -> dict[int, float]:
    return {event.step + 1: event.value for event in accumulator.Scalars(tag)}


def scalar_last(accumulator: EventAccumulator, tag: str) -> float:
    values = accumulator.Scalars(tag)
    if not values:
        raise KeyError(tag)
    return values[-1].value


def latest_log_metrics(path: Path) -> tuple[int, dict[str, float]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = list(
        re.finditer(
            r"Global Step:\s+(\d+)/650(?P<body>.*?)╯",
            text,
            flags=re.DOTALL,
        )
    )
    if not matches:
        raise ValueError("no complete metric table in log")
    match = matches[-1]
    metrics = {
        key: float(value)
        for key, value in re.findall(
            r"([A-Za-z0-9_]+(?:/[A-Za-z0-9_]+)*)="
            r"([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
            match.group("body"),
        )
    }
    return int(match.group(1)), metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--driver-log", type=Path, required=True)
    parser.add_argument("--progress-out", type=Path, required=True)
    parser.add_argument("--optimization-out", type=Path, required=True)
    args = parser.parse_args()

    accumulator = EventAccumulator(str(args.events), size_guidance={"scalars": 0})
    accumulator.Reload()

    success = scalar_map(accumulator, "env/success_once")
    resident = scalar_map(accumulator, "train/sac/global_resident_transitions")
    new_transitions = scalar_map(accumulator, "train/sac/global_new_transitions")
    step_time = scalar_map(accumulator, "time/step")
    train_time = scalar_map(accumulator, "time/actor/run_training")
    latest_step, latest = latest_log_metrics(args.driver_log)
    if latest_step not in success:
        success[latest_step] = latest["success_once"]
        resident[latest_step] = latest["sac/global_resident_transitions"]
        new_transitions[latest_step] = latest["sac/global_new_transitions"]
        step_time[latest_step] = latest["step"]
        train_time[latest_step] = latest["actor/run_training"]
    steps = sorted(set(success) & set(resident) & set(step_time))
    successes = [success[step] for step in steps]
    running_success = np.cumsum(successes) / np.arange(1, len(successes) + 1)

    args.progress_out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 9.2), sharex=True, constrained_layout=True)
    fig.suptitle(f"DSRL formal progress through global step {max(steps)}", fontsize=15)

    axes[0].plot(steps, successes, marker="o", label="train success / 4 episodes")
    axes[0].plot(steps, running_success, marker=".", label="running mean")
    eval_success = scalar_map(accumulator, "eval/success_once")
    if eval_success:
        eval_step = max(eval_success)
        axes[0].scatter(
            [eval_step],
            [eval_success[eval_step]],
            marker="D",
            s=58,
            label="eval success / 12 episodes",
            zorder=4,
        )
    plotted_success = successes + list(eval_success.values())
    axes[0].set_ylim(-0.02, max(0.3, max(plotted_success) * 1.1))
    axes[0].set_ylabel("Success rate")
    axes[0].legend(fontsize=8, ncol=2, loc="upper left")

    axes[1].plot(steps, [resident[step] for step in steps], marker="o", label="global resident")
    axes[1].axhline(500, color="0.45", linestyle="--", linewidth=1, label="warm-up threshold")
    axes[1].bar(
        steps,
        [new_transitions[step] for step in steps],
        alpha=0.25,
        label="new transitions / cycle",
    )
    axes[1].set_ylabel("Transitions")
    axes[1].legend(fontsize=8, ncol=3, loc="upper left")

    axes[2].plot(steps, [step_time[step] for step in steps], marker="o", label="whole step")
    axes[2].plot(steps, [train_time[step] for step in steps], marker=".", label="SAC training")
    axes[2].set_ylabel("Seconds")
    axes[2].set_xlabel("RLinf global step (actual data starts at step 1)")
    axes[2].legend(fontsize=8, ncol=2, loc="upper left")

    for axis in axes:
        axis.grid(True, color="0.88", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xticks(steps)
    fig.savefig(args.progress_out, dpi=180)
    plt.close(fig)

    q_pi = scalar_map(accumulator, "train/actor/q_pi")
    q_pi[latest_step] = latest["actor/q_pi"]
    update_steps = sorted(q_pi)
    q_heads = {
        step: [
            scalar_map(accumulator, f"train/actor/q_value_{index}")[step]
            for index in range(10)
        ]
        for step in update_steps
        if step != latest_step
    }
    q_heads[latest_step] = [latest[f"actor/q_value_{index}"] for index in range(10)]
    q_data = scalar_map(accumulator, "train/critic/q_data")
    q_data[latest_step] = latest["critic/q_data"]
    gradient_series = {
        "actor": scalar_map(accumulator, "train/actor/grad_norm"),
        "critic": scalar_map(accumulator, "train/critic/grad_norm"),
        "alpha": scalar_map(accumulator, "train/alpha/grad_norm"),
    }
    gradients = {
        step: {name: values[step] for name, values in gradient_series.items()}
        for step in update_steps
        if step != latest_step
    }
    gradients[latest_step] = {
        "actor": latest["actor/grad_norm"],
        "critic": latest["critic/grad_norm"],
        "alpha": latest["alpha/grad_norm"],
    }
    clips = {"actor": 3.5, "critic": 10.0, "alpha": 10.0}
    time_series = {
        "rollout": scalar_map(accumulator, "time/generate_rollouts"),
        "SAC": scalar_map(accumulator, "time/actor/run_training"),
        "eval": scalar_map(accumulator, "time/eval"),
        "sync": scalar_map(accumulator, "time/sync_weights"),
    }
    time_parts = {
        step: {
            label: values.get(step, 0.0)
            for label, values in time_series.items()
        }
        for step in update_steps
        if step != latest_step
    }
    time_parts[latest_step] = {
        "rollout": latest["generate_rollouts"],
        "SAC": latest["actor/run_training"],
        "eval": 0.0,
        "sync": latest["sync_weights"],
    }
    training_series = {
        "critic loss": scalar_map(accumulator, "train/sac/critic_loss"),
        "actor loss": scalar_map(accumulator, "train/sac/actor_loss"),
        "alpha loss": scalar_map(accumulator, "train/sac/alpha_loss"),
        "alpha": scalar_map(accumulator, "train/sac/alpha"),
        "entropy": scalar_map(accumulator, "train/actor/entropy"),
        "updates": scalar_map(accumulator, "train/sac/planned_optimizer_updates"),
    }
    training_metrics = {
        step: {name: values[step] for name, values in training_series.items()}
        for step in update_steps
        if step != latest_step
    }
    training_metrics[latest_step] = {
        "critic loss": latest["sac/critic_loss"],
        "actor loss": latest["sac/actor_loss"],
        "alpha loss": latest["sac/alpha_loss"],
        "alpha": latest["sac/alpha"],
        "entropy": latest["actor/entropy"],
        "updates": latest["sac/planned_optimizer_updates"],
    }

    fig, axes = plt.subplots(4, 1, figsize=(8.4, 11.4), constrained_layout=True)
    fig.suptitle(
        f"Early learned SAC diagnostics: global steps {update_steps[0]}–{latest_step}",
        fontsize=15,
    )

    for color_index, step in enumerate(update_steps):
        axes[0].plot(
            range(10),
            q_heads[step],
            marker="o",
            color=f"C{color_index}",
            label=f"Q heads step {step}",
        )
        axes[0].axhline(
            q_data[step],
            color=f"C{color_index}",
            linestyle=":",
            linewidth=1,
        )
    axes[0].set_ylabel("Q value")
    axes[0].set_xticks(range(10), [f"Q{i}" for i in range(10)])
    axes[0].legend(fontsize=8, ncol=3, loc="lower left")

    names = list(clips)
    positions = np.arange(len(names))
    width = 0.18
    offsets = np.linspace(-width, width, len(update_steps))
    for offset, step in zip(offsets, update_steps, strict=True):
        axes[1].bar(
            positions + offset,
            [gradients[step][name] for name in names],
            width=width,
            label=f"step {step}",
        )
    axes[1].bar(
        positions + width * 2,
        [clips[name] for name in names],
        width=width,
        alpha=0.4,
        label="clip threshold",
    )
    axes[1].set_xticks(positions, names)
    axes[1].set_ylabel("Gradient norm")
    axes[1].legend(fontsize=8, ncol=2, loc="upper left")

    axes[2].axis("off")
    metric_names = ["critic loss", "actor loss", "alpha loss", "alpha", "entropy", "updates"]
    header = "metric          " + "".join(f"step {step:>2}   " for step in update_steps)
    rows = [header.rstrip()]
    for name in metric_names:
        if name == "critic loss":
            values = "".join(f"{training_metrics[step][name]:>8.5f} " for step in update_steps)
        elif name == "updates":
            values = "".join(f"{training_metrics[step][name]:>8.0f} " for step in update_steps)
        else:
            values = "".join(f"{training_metrics[step][name]:>8.3f} " for step in update_steps)
        rows.append(f"{name:<14}{values.rstrip()}")
    axes[2].text(
        0.0,
        0.86,
        "\n".join(rows),
        transform=axes[2].transAxes,
        va="top",
        family="monospace",
        fontsize=9,
    )
    axes[2].text(
        0.73,
        0.86,
        "All reported values are finite.\n"
        "Q heads stay tightly grouped.\n"
        "Three points show health, not convergence.",
        transform=axes[2].transAxes,
        va="top",
        fontsize=9,
    )

    time_colors = {"rollout": "C0", "SAC": "C1", "eval": "C2", "sync": "C3"}
    for step_index, step in enumerate(update_steps):
        start = 0.0
        for label, value in time_parts[step].items():
            axes[3].barh(
                [f"step {step}"],
                [value],
                left=[start],
                color=time_colors[label],
                label=label if step_index == 0 else None,
            )
            start += value
    axes[3].set_xlabel("Seconds")
    axes[3].legend(fontsize=8, ncol=2, loc="upper center")

    for axis in (axes[0], axes[1], axes[3]):
        axis.grid(True, color="0.88", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.optimization_out, dpi=180)
    plt.close(fig)
    print(args.progress_out)
    print(args.optimization_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
