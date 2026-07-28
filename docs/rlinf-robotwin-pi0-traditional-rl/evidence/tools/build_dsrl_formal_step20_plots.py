#!/usr/bin/env python3
"""Build DSRL formal success, optimization, timing, and summary artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def scalar_map(accumulator: EventAccumulator, tag: str) -> dict[int, float]:
    return {event.step + 1: float(event.value) for event in accumulator.Scalars(tag)}


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z**2 / (4 * total**2)
        )
        / denominator
    )
    return center - radius, center + radius


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--success-out", type=Path, required=True)
    parser.add_argument("--optimization-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--timing-csv-out", type=Path, required=True)
    parser.add_argument(
        "--layout",
        choices=("portrait", "landscape"),
        default="portrait",
        help="Use landscape for readable in-conversation previews.",
    )
    args = parser.parse_args()

    accumulator = EventAccumulator(str(args.events), size_guidance={"scalars": 0})
    accumulator.Reload()

    success = scalar_map(accumulator, "env/success_once")
    train_return = scalar_map(accumulator, "env/return")
    new_transitions = scalar_map(
        accumulator, "train/sac/global_new_transitions"
    )
    resident = scalar_map(accumulator, "train/sac/global_resident_transitions")
    planned_updates = scalar_map(
        accumulator, "train/sac/planned_optimizer_updates"
    )
    step_time = scalar_map(accumulator, "time/step")
    rollout_time = scalar_map(accumulator, "time/generate_rollouts")
    training_time = scalar_map(accumulator, "time/actor/run_training")
    eval_time = scalar_map(accumulator, "time/eval")
    sync_time = scalar_map(accumulator, "time/sync_weights")
    eval_success = scalar_map(accumulator, "eval/success_once")

    steps = sorted(set(success) & set(resident) & set(step_time))
    latest_step = max(steps)
    if steps != list(range(1, latest_step + 1)):
        raise ValueError(f"non-contiguous global steps: {steps}")

    train_successes = np.array([success[step] for step in steps])
    interactions = np.array([resident[step] * 20 for step in steps])
    cumulative_updates = np.cumsum(
        [planned_updates.get(step, 0.0) for step in steps]
    )
    rolling_cycles = 5
    rolling_success = np.array(
        [
            np.mean(train_successes[max(0, index - rolling_cycles + 1) : index + 1])
            for index in range(len(steps))
        ]
    )

    gaussian_steps = [step for step in steps if step <= 13]
    learned_steps = [step for step in steps if step >= 14]
    gaussian_successes = round(sum(success[step] * 4 for step in gaussian_steps))
    gaussian_episodes = len(gaussian_steps) * 4
    learned_successes = round(sum(success[step] * 4 for step in learned_steps))
    learned_episodes = len(learned_steps) * 4

    eval_points = []
    for step, rate in sorted(eval_success.items()):
        count = round(rate * 12)
        lower, upper = wilson_interval(count, 12)
        eval_points.append(
            {
                "step": step,
                "successes": count,
                "episodes": 12,
                "rate": rate,
                "wilson95_low": lower,
                "wilson95_high": upper,
                "requested_interactions": resident[step] * 20,
            }
        )

    args.success_out.parent.mkdir(parents=True, exist_ok=True)
    if args.layout == "landscape":
        fig = plt.figure(figsize=(13.5, 8.0), constrained_layout=True)
        grid = fig.add_gridspec(2, 2, height_ratios=(1.18, 1.0))
        axes = [
            fig.add_subplot(grid[0, :]),
            fig.add_subplot(grid[1, 0]),
            fig.add_subplot(grid[1, 1]),
        ]
    else:
        fig, axes = plt.subplots(
            3, 1, figsize=(7.4, 10.8), constrained_layout=True
        )
    fig.suptitle(
        f"DSRL formal success and sample budget through global step {latest_step}",
        fontsize=15,
    )

    interaction_k = interactions / 1000
    axes[0].plot(
        interaction_k,
        train_successes,
        marker="o",
        linewidth=1.4,
        label="train: 4 episodes / cycle",
    )
    axes[0].plot(
        interaction_k,
        rolling_success,
        marker=".",
        linewidth=2,
        label="train: trailing 20 episodes",
    )
    warmup_x = resident[13] * 20 / 1000
    axes[0].axvline(
        warmup_x,
        color="0.45",
        linestyle="--",
        linewidth=1,
        label="learned rollout starts next cycle",
    )
    gaussian_rate = gaussian_successes / gaussian_episodes
    learned_rate = learned_successes / learned_episodes
    axes[0].hlines(
        gaussian_rate,
        interaction_k[0],
        interaction_k[12],
        colors="C3",
        linestyles=":",
        label=f"Gaussian phase mean: {gaussian_successes}/{gaussian_episodes}",
    )
    axes[0].hlines(
        learned_rate,
        interaction_k[13],
        interaction_k[-1],
        colors="C4",
        linestyles=":",
        label=f"learned phase mean: {learned_successes}/{learned_episodes}",
    )
    for point in eval_points:
        x = point["requested_interactions"] / 1000
        y = point["rate"]
        axes[0].errorbar(
            [x],
            [y],
            yerr=[
                [y - point["wilson95_low"]],
                [point["wilson95_high"] - y],
            ],
            fmt="D",
            color="C2",
            capsize=4,
            markersize=7,
            label="formal eval: 12 episodes"
            if point is eval_points[0]
            else None,
            zorder=5,
        )
    axes[0].set_xlim(left=0)
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("Success rate")
    axes[0].set_xlabel("Cumulative requested primitive interactions (thousands)")
    axes[0].legend(fontsize=8, ncol=2, loc="upper left")

    axes[1].plot(
        steps,
        interactions / 1000,
        marker="o",
        label="requested primitive interactions",
    )
    axes[1].plot(
        steps,
        cumulative_updates / 1000,
        marker=".",
        label="optimizer updates",
    )
    axes[1].axvline(13, color="0.45", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Cumulative count (thousands)")
    axes[1].set_xlabel("RLinf global step")
    axes[1].legend(fontsize=8, ncol=2, loc="upper left")

    axes[2].plot(
        steps,
        [step_time[step] for step in steps],
        marker="o",
        label="whole cycle",
    )
    axes[2].plot(
        steps,
        [training_time[step] for step in steps],
        marker=".",
        label="SAC training",
    )
    axes[2].plot(
        steps,
        [rollout_time[step] for step in steps],
        marker=".",
        label="rollout",
    )
    if eval_time:
        axes[2].scatter(
            list(eval_time),
            list(eval_time.values()),
            marker="D",
            s=55,
            label="eval overhead",
            zorder=4,
        )
    axes[2].set_ylabel("Seconds")
    axes[2].set_xlabel("RLinf global step")
    axes[2].legend(fontsize=8, ncol=2, loc="upper left")

    for axis in axes:
        axis.grid(True, color="0.88", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9)
    fig.savefig(args.success_out, dpi=180)
    plt.close(fig)

    q_pi = scalar_map(accumulator, "train/actor/q_pi")
    q_data = scalar_map(accumulator, "train/critic/q_data")
    critic_loss = scalar_map(accumulator, "train/sac/critic_loss")
    actor_loss = scalar_map(accumulator, "train/sac/actor_loss")
    alpha_loss = scalar_map(accumulator, "train/sac/alpha_loss")
    alpha = scalar_map(accumulator, "train/sac/alpha")
    entropy = scalar_map(accumulator, "train/actor/entropy")
    actor_grad = scalar_map(accumulator, "train/actor/grad_norm")
    critic_grad = scalar_map(accumulator, "train/critic/grad_norm")
    alpha_grad = scalar_map(accumulator, "train/alpha/grad_norm")
    update_steps = sorted(q_pi)
    q_heads = [
        scalar_map(accumulator, f"train/actor/q_value_{index}")
        for index in range(10)
    ]
    q_min = [min(head[step] for head in q_heads) for step in update_steps]
    q_max = [max(head[step] for head in q_heads) for step in update_steps]

    if args.layout == "landscape":
        fig = plt.figure(figsize=(13.5, 8.0), constrained_layout=True)
        grid = fig.add_gridspec(2, 3)
        axes = [
            fig.add_subplot(grid[0, 0:2]),
            fig.add_subplot(grid[0, 2]),
            fig.add_subplot(grid[1, 0]),
            fig.add_subplot(grid[1, 1]),
            fig.add_subplot(grid[1, 2]),
        ]
    else:
        fig, axes = plt.subplots(
            5, 1, figsize=(7.4, 13.6), sharex=True, constrained_layout=True
        )
    fig.suptitle(
        f"DSRL learned SAC diagnostics through global step {latest_step}",
        fontsize=15,
    )

    axes[0].fill_between(
        update_steps,
        q_min,
        q_max,
        alpha=0.2,
        label="10-Q head range",
    )
    axes[0].plot(update_steps, [q_pi[step] for step in update_steps], marker="o", label="Qπ")
    axes[0].plot(
        update_steps,
        [q_data[step] for step in update_steps],
        marker=".",
        label="Qdata",
    )
    axes[0].set_ylabel("Q value")
    axes[0].legend(fontsize=8, ncol=3, loc="lower left")

    axes[1].plot(
        update_steps,
        [actor_loss[step] for step in update_steps],
        marker="o",
        label="actor loss",
    )
    axes[1].plot(
        update_steps,
        [alpha_loss[step] for step in update_steps],
        marker=".",
        label="alpha loss",
    )
    axes[1].axhline(0, color="0.5", linewidth=0.8)
    axes[1].set_ylabel("Actor / alpha loss")
    axes[1].legend(fontsize=8, ncol=2, loc="upper right")

    axes[2].plot(
        update_steps,
        [critic_loss[step] for step in update_steps],
        marker="o",
        label="critic loss",
    )
    axes[2].set_ylabel("Critic loss")
    axes[2].legend(fontsize=8, loc="upper left")

    axes[3].plot(
        update_steps,
        [actor_grad[step] for step in update_steps],
        marker="o",
        label="actor grad",
    )
    axes[3].plot(
        update_steps,
        [critic_grad[step] for step in update_steps],
        marker="o",
        label="critic grad",
    )
    axes[3].plot(
        update_steps,
        [alpha_grad[step] for step in update_steps],
        marker="o",
        label="alpha grad",
    )
    axes[3].axhline(3.5, color="C0", linestyle=":", linewidth=1, label="actor clip")
    axes[3].axhline(10.0, color="0.45", linestyle=":", linewidth=1, label="critic/alpha clip")
    axes[3].set_ylabel("Pre-clip grad norm")
    axes[3].legend(fontsize=8, ncol=3, loc="upper right")

    axes[4].plot(
        update_steps,
        [alpha[step] for step in update_steps],
        marker="o",
        label="alpha",
    )
    entropy_axis = axes[4].twinx()
    entropy_axis.plot(
        update_steps,
        [entropy[step] for step in update_steps],
        marker=".",
        color="C1",
        label="entropy",
    )
    axes[4].set_ylabel("Alpha")
    entropy_axis.set_ylabel("Entropy")
    axes[4].set_xlabel("RLinf global step")
    handles_left, labels_left = axes[4].get_legend_handles_labels()
    handles_right, labels_right = entropy_axis.get_legend_handles_labels()
    axes[4].legend(
        handles_left + handles_right,
        labels_left + labels_right,
        fontsize=8,
        ncol=2,
        loc="upper right",
    )

    for axis in axes:
        axis.grid(True, color="0.88", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    entropy_axis.spines["top"].set_visible(False)
    entropy_axis.tick_params(labelsize=9)
    if args.layout == "portrait":
        axes[-1].set_xticks(update_steps)
    fig.savefig(args.optimization_out, dpi=180)
    plt.close(fig)

    total_training_seconds = sum(training_time[step] for step in update_steps)
    total_updates = sum(planned_updates[step] for step in update_steps)
    learned_cycle_seconds = [step_time[step] for step in learned_steps]
    summary = {
        "latest_step": latest_step,
        "resident_transitions": resident[latest_step],
        "requested_primitive_interactions": interactions[-1],
        "optimizer_updates": total_updates,
        "gaussian": {
            "steps": gaussian_steps,
            "successes": gaussian_successes,
            "episodes": gaussian_episodes,
            "rate": gaussian_rate,
        },
        "learned": {
            "steps": learned_steps,
            "successes": learned_successes,
            "episodes": learned_episodes,
            "rate": learned_rate,
        },
        "formal_eval": eval_points,
        "next_eval_step": (latest_step // 13 + 1) * 13,
        "updates_per_second": total_updates / total_training_seconds,
        "learned_cycle_seconds_mean": float(np.mean(learned_cycle_seconds)),
        "latest": {
            "train_success": success[latest_step],
            "train_return": train_return[latest_step],
            "new_transitions": new_transitions[latest_step],
            "planned_updates": planned_updates[latest_step],
            "step_seconds": step_time[latest_step],
            "training_seconds": training_time[latest_step],
            "critic_loss": critic_loss[latest_step],
            "actor_loss": actor_loss[latest_step],
            "alpha_loss": alpha_loss[latest_step],
            "alpha": alpha[latest_step],
            "entropy": entropy[latest_step],
            "q_pi": q_pi[latest_step],
            "q_data": q_data[latest_step],
            "q_head_min": q_min[-1],
            "q_head_max": q_max[-1],
            "actor_grad": actor_grad[latest_step],
            "critic_grad": critic_grad[latest_step],
            "alpha_grad": alpha_grad[latest_step],
        },
        "series": {
            "step": steps,
            "requested_interactions": interactions.tolist(),
            "train_success": train_successes.tolist(),
            "train_success_trailing_20_episodes": rolling_success.tolist(),
            "resident_transitions": [resident[step] for step in steps],
            "new_transitions": [new_transitions[step] for step in steps],
            "cumulative_optimizer_updates": cumulative_updates.tolist(),
        },
        "optimization_series": {
            "step": update_steps,
            "q_pi": [q_pi[step] for step in update_steps],
            "q_data": [q_data[step] for step in update_steps],
            "q_head_min": q_min,
            "q_head_max": q_max,
            "critic_loss": [critic_loss[step] for step in update_steps],
            "actor_loss": [actor_loss[step] for step in update_steps],
            "alpha_loss": [alpha_loss[step] for step in update_steps],
            "alpha": [alpha[step] for step in update_steps],
            "entropy": [entropy[step] for step in update_steps],
            "actor_grad": [actor_grad[step] for step in update_steps],
            "critic_grad": [critic_grad[step] for step in update_steps],
            "alpha_grad": [alpha_grad[step] for step in update_steps],
        },
    }
    args.summary_out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.timing_csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.timing_csv_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "global_step",
                "phase",
                "train_success_rate",
                "new_transitions",
                "resident_transitions",
                "requested_primitive_interactions",
                "planned_optimizer_updates",
                "rollout_seconds",
                "sac_training_seconds",
                "eval_seconds",
                "sync_seconds",
                "whole_step_seconds",
            ],
        )
        writer.writeheader()
        for step in steps:
            if step <= 12:
                phase = "gaussian_warmup"
            elif step == 13:
                phase = "threshold_update_eval"
            else:
                phase = "learned"
            writer.writerow(
                {
                    "global_step": step,
                    "phase": phase,
                    "train_success_rate": success[step],
                    "new_transitions": new_transitions[step],
                    "resident_transitions": resident[step],
                    "requested_primitive_interactions": resident[step] * 20,
                    "planned_optimizer_updates": planned_updates.get(step, 0.0),
                    "rollout_seconds": rollout_time[step],
                    "sac_training_seconds": training_time[step],
                    "eval_seconds": eval_time.get(step, 0.0),
                    "sync_seconds": sync_time.get(step, 0.0),
                    "whole_step_seconds": step_time[step],
                }
            )
    print(args.success_out)
    print(args.optimization_out)
    print(args.summary_out)
    print(args.timing_csv_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
