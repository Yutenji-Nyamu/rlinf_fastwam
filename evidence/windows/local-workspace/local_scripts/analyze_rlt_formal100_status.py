"""Analyze the completed RoboTwin RLT Stage 2 formal 100-cycle pilot."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


SCALAR_TAGS = [
    "env/success_once",
    "env/return",
    "eval/success_once",
    "eval/success_at_end",
    "train/replay/actor_switch_rate",
    "train/rlt/ready_for_online",
    "train/rlt/update_step",
    "train/rlt/desired_total_updates",
    "train/rlt/pending_update_budget",
    "train/actor/actor_weight_ramp_progress",
    "train/actor/bc_loss",
    "train/actor/bc_weight",
    "train/actor/q_weight",
    "train/actor/q_value_0",
    "train/actor/q_value_1",
    "train/actor/q_pi",
    "train/critic/q_data",
    "train/actor/grad_norm",
    "train/critic/grad_norm",
    "train/sac/actor_loss",
    "train/sac/critic_loss",
    "time/step",
]

CHECKPOINT_BYTES = {
    10: 69_830_237,
    20: 88_122_958,
    30: 107_036_130,
    40: 125_769_309,
    50: 144_478_832,
    60: 163_259_314,
    70: 181_968_837,
    80: 200_843_931,
    90: 218_725_599,
    100: 237_293_208,
}
CHECKPOINT_TRAJECTORY_BYTES = {
    10: 17_485_072,
    20: 35_777_788,
    30: 54_690_954,
    40: 73_424_130,
    50: 92_133_653,
    60: 110_914_135,
    70: 129_623_658,
    80: 148_498_752,
    90: 166_380_420,
    100: 184_948_025,
}
CHECKPOINT_TRAJECTORY_FILES = {
    10: 742,
    20: 1516,
    30: 2316,
    40: 3108,
    50: 3899,
    60: 4693,
    70: 5484,
    80: 6282,
    90: 7038,
    100: 7823,
}


def scalar_frame(event_path: Path) -> tuple[EventAccumulator, pd.DataFrame]:
    accumulator = EventAccumulator(
        str(event_path),
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()
    rows: list[dict[str, float | int | str]] = []
    available = set(accumulator.Tags()["scalars"])
    for tag in SCALAR_TAGS:
        if tag not in available:
            continue
        for event in accumulator.Scalars(tag):
            rows.append(
                {
                    "tag": tag,
                    "cycle": event.step + 1,
                    "value": float(event.value),
                    "wall_time": float(event.wall_time),
                }
            )
    return accumulator, pd.DataFrame(rows)


def values(accumulator: EventAccumulator, tag: str) -> pd.Series:
    return pd.Series(
        {event.step + 1: float(event.value) for event in accumulator.Scalars(tag)},
        dtype=float,
    ).sort_index()


def first_cycle(series: pd.Series, threshold: float = 0.5) -> int:
    return int(series[series >= threshold].index[0])


def period_success(
    success: pd.Series,
    cycles: list[int],
    episodes_per_cycle: int = 4,
) -> dict[str, float | int]:
    selected = success.loc[cycles]
    episodes = len(selected) * episodes_per_cycle
    successes = float(selected.sum() * episodes_per_cycle)
    return {
        "cycles": len(selected),
        "episodes": episodes,
        "successes": successes,
        "rate": float(selected.mean()),
    }


def metric_stats(series: pd.Series) -> dict[str, float]:
    return {
        "count": int(series.size),
        "min": float(series.min()),
        "max": float(series.max()),
        "first": float(series.iloc[0]),
        "last": float(series.iloc[-1]),
        "first10_mean": float(series.iloc[:10].mean()),
        "last10_mean": float(series.iloc[-10:].mean()),
    }


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_success(
    output: Path,
    success: pd.Series,
    eval_success: pd.Series,
    switch_cycle: int,
    ramp_cycle: int,
    summary: dict,
) -> None:
    configure_plotting()
    rolling = success.rolling(10, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        success.index,
        success.values * 100,
        s=18,
        alpha=0.35,
        label="Train success / cycle (4 episodes)",
    )
    ax.plot(
        rolling.index,
        rolling.values * 100,
        linewidth=2.4,
        label="Train rolling 10-cycle success",
    )
    ax.plot(
        eval_success.index,
        eval_success.values * 100,
        marker="D",
        markersize=6,
        linewidth=1.8,
        label="Deterministic eval (4 episodes)",
    )
    ax.axvspan(1, switch_cycle - 0.5, alpha=0.08, label="Reference control")
    ax.axvline(switch_cycle, color="tab:orange", linewidth=1.5)
    ax.axvline(ramp_cycle, color="tab:red", linewidth=1.5, linestyle="--")
    ax.annotate(
        f"Student starts: cycle {switch_cycle}",
        (switch_cycle, 54),
        xytext=(switch_cycle + 2, 54),
    )
    ax.annotate(
        f"BC/Q ramp complete: cycle {ramp_cycle}",
        (ramp_cycle, 47),
        xytext=(ramp_cycle + 2, 47),
    )
    ref = summary["train_reference"]
    student = summary["train_student"]
    tail = summary["train_student_last20"]
    ax.text(
        0.02,
        0.97,
        (
            f"Reference: {ref['successes']:.0f}/{ref['episodes']} "
            f"({100 * ref['rate']:.1f}%)\n"
            f"Student overall: {student['successes']:.0f}/{student['episodes']} "
            f"({100 * student['rate']:.1f}%)\n"
            f"Student last 20 cycles: {tail['successes']:.0f}/{tail['episodes']} "
            f"({100 * tail['rate']:.1f}%)"
        ),
        transform=ax.transAxes,
        va="top",
    )
    ax.set(
        title="RoboTwin RLT formal pilot: success and controller phases",
        xlabel="Outer cycle",
        ylabel="Success rate (%)",
        xlim=(1, 100),
        ylim=(-2, 58),
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_optimization(
    output: Path,
    accumulator: EventAccumulator,
    switch_cycle: int,
    ramp_cycle: int,
) -> None:
    configure_plotting()
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    actor_loss = values(accumulator, "train/sac/actor_loss")
    bc_loss = values(accumulator, "train/actor/bc_loss")
    critic_loss = values(accumulator, "train/sac/critic_loss")
    axes[0].semilogy(
        actor_loss.index,
        np.maximum(actor_loss.values, 1e-7),
        label="Actor loss",
    )
    axes[0].semilogy(
        bc_loss.index,
        np.maximum(bc_loss.values, 1e-7),
        label="BC loss",
    )
    axes[0].semilogy(
        critic_loss.index,
        np.maximum(critic_loss.values, 1e-7),
        label="Critic loss",
    )
    axes[0].set_ylabel("Loss (log scale)")
    axes[0].legend(ncol=3)

    for tag, label in [
        ("train/actor/q_value_0", "Q0(actor action)"),
        ("train/actor/q_value_1", "Q1(actor action)"),
        ("train/critic/q_data", "Q(data)"),
    ]:
        series = values(accumulator, tag)
        axes[1].plot(series.index, series.values, label=label)
    axes[1].set_ylabel("Q value")
    axes[1].legend(ncol=3)

    actor_grad = values(accumulator, "train/actor/grad_norm")
    critic_grad = values(accumulator, "train/critic/grad_norm")
    axes[2].plot(actor_grad.index, actor_grad.values, label="Actor grad norm")
    axes[2].plot(critic_grad.index, critic_grad.values, label="Critic grad norm")
    axes[2].axhline(10, color="tab:red", linestyle="--", label="Clip threshold")
    axes[2].set(ylabel="Global grad norm", xlabel="Outer cycle", ylim=(0, 10.5))
    axes[2].legend(ncol=3)

    for ax in axes:
        ax.axvline(switch_cycle, color="tab:orange", linewidth=1.2)
        ax.axvline(ramp_cycle, color="tab:red", linewidth=1.2, linestyle="--")
        ax.set_xlim(1, 100)
    axes[0].set_title("Optimization health (finite losses and gradients below clip=10)")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_resources(output: Path, resource: pd.DataFrame) -> None:
    configure_plotting()
    frame = resource.copy()
    frame["minute"] = (frame["unix_time"] - frame["unix_time"].iloc[0]) / 60
    frame["gpu0_util_roll"] = frame["gpu0_util_pct"].rolling(30, min_periods=1).mean()
    frame["gpu1_util_roll"] = frame["gpu1_util_pct"].rolling(30, min_periods=1).mean()

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    axes[0].plot(frame["minute"], frame["gpu0_used_mib"] / 1024, label="GPU0 memory")
    axes[0].plot(frame["minute"], frame["gpu1_used_mib"] / 1024, label="GPU1 memory")
    axes[0].set_ylabel("GPU memory (GiB)")
    util = axes[0].twinx()
    util.plot(
        frame["minute"],
        frame["gpu0_util_roll"],
        alpha=0.55,
        linewidth=1,
        label="GPU0 util, 60s mean",
    )
    util.plot(
        frame["minute"],
        frame["gpu1_util_roll"],
        alpha=0.55,
        linewidth=1,
        label="GPU1 util, 60s mean",
    )
    util.set_ylabel("GPU utilization (%)")
    lines, labels = axes[0].get_legend_handles_labels()
    lines2, labels2 = util.get_legend_handles_labels()
    axes[0].legend(lines + lines2, labels + labels2, ncol=2, loc="upper right")

    axes[1].plot(
        frame["minute"],
        frame["cgroup_anon_bytes"] / 2**30,
        label="cgroup anon",
    )
    axes[1].plot(
        frame["minute"],
        frame["matched_total_rss_kib"] / 2**20,
        label="Matched process RSS",
    )
    axes[1].plot(
        frame["minute"],
        frame["env_rss_kib"] / 2**20,
        label="Env-worker RSS",
    )
    axes[1].plot(
        frame["minute"],
        frame["rollout_rss_kib"] / 2**20,
        label="Rollout-worker RSS",
    )
    axes[1].set_ylabel("Resident / anon memory (GiB)")
    axes[1].legend(ncol=2)

    axes[2].plot(
        frame["minute"],
        frame["cgroup_current_bytes"] / 2**30,
        label="cgroup current",
    )
    axes[2].plot(
        frame["minute"],
        frame["cgroup_file_bytes"] / 2**30,
        label="cgroup file cache",
    )
    axes[2].axhline(240, color="tab:red", linestyle="--", label="memory.max 240 GiB")
    axes[2].set(
        ylabel="Cgroup accounted memory (GiB)",
        xlabel="Minutes since launch",
    )
    axes[2].legend(ncol=3)
    axes[0].set_title("Resource profile (no OOM; cgroup limit pressure was file-cache dominated)")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_artifacts(output: Path) -> None:
    configure_plotting()
    steps = np.array(sorted(CHECKPOINT_BYTES))
    total = np.array([CHECKPOINT_BYTES[x] for x in steps]) / 2**20
    replay = np.array([CHECKPOINT_TRAJECTORY_BYTES[x] for x in steps]) / 2**20
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(steps, total, width=6, alpha=0.7, label="Checkpoint total")
    ax.bar(steps, replay, width=6, alpha=0.8, label="Replay trajectory payload")
    for step, size, count in zip(
        steps,
        total,
        [CHECKPOINT_TRAJECTORY_FILES[x] for x in steps],
    ):
        ax.text(step, size + 3, f"{size:.0f} MiB\n{count:,} traj", ha="center", fontsize=8)
    ax.set(
        title="Checkpoint growth is linear and replay-dominated",
        xlabel="Checkpoint cycle",
        ylabel="Size (MiB)",
        xticks=steps,
        ylim=(0, total.max() * 1.18),
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("status_dir")
    args = parser.parse_args()
    status_dir = Path(args.status_dir)
    event_path = next(status_dir.glob("events.out.tfevents.*"))
    accumulator, scalar_long = scalar_frame(event_path)
    resource = pd.read_csv(status_dir / "resources.csv")

    success = values(accumulator, "env/success_once")
    eval_success = values(accumulator, "eval/success_once")
    switch = values(accumulator, "train/replay/actor_switch_rate")
    ready = values(accumulator, "train/rlt/ready_for_online")
    ramp = values(accumulator, "train/actor/actor_weight_ramp_progress")
    switch_cycle = first_cycle(switch)
    ready_cycle = first_cycle(ready)
    ramp_cycle = first_cycle(ramp, 0.999)

    reference_cycles = [int(x) for x in success.index if switch.loc[x] < 0.5]
    student_cycles = [int(x) for x in success.index if switch.loc[x] >= 0.5]
    eval_cycles = [int(x) for x in eval_success.index]
    state_files = sorted(status_dir.glob("rlt_state_rank_*_step100.pt"))
    states = [
        torch.load(path, map_location="cpu", weights_only=False) for path in state_files
    ]

    started = datetime.fromisoformat("2026-07-30T01:15:54+08:00")
    finished = datetime.fromisoformat(
        (status_dir / "finished_at.txt").read_text().strip()
    )
    active = resource[
        (resource["compute_process_count"] > 0)
        & (resource["gpu0_used_mib"] > 1000)
        & (resource["gpu1_used_mib"] > 1000)
    ]
    summary = {
        "run": {
            "environment": "robotwin",
            "task": "adjust_bottle",
            "cycles": 100,
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "wall_clock_seconds": int((finished - started).total_seconds()),
            "exit_code": int((status_dir / "exit_code.txt").read_text().strip()),
            "gpu_hours_reserved": 2 * (finished - started).total_seconds() / 3600,
        },
        "phase": {
            "warmup_anchor_episodes": states[0][
                "global_warmup_ready_total_episodes"
            ],
            "warmup_anchor_transitions": states[0][
                "global_warmup_ready_total_transitions"
            ],
            "ready_cycle": ready_cycle,
            "student_switch_cycle": switch_cycle,
            "full_ramp_cycle": ramp_cycle,
            "final_update_step": states[0]["update_step"],
            "final_desired_updates": float(
                values(accumulator, "train/rlt/desired_total_updates").iloc[-1]
            ),
            "final_pending_update_budget": float(
                values(accumulator, "train/rlt/pending_update_budget").iloc[-1]
            ),
            "critic_optimizer_updates": states[0]["update_step"],
            "actor_optimizer_updates": int(
                values(accumulator, "train/rlt/actor_updates_run").sum()
            ),
            "macro_transitions": sum(
                state["local_total_transitions_added"] for state in states
            ),
            "train_episodes": sum(
                state["local_total_episodes_added"] for state in states
            ),
            "per_rank_replay": [
                state["local_total_transitions_added"] for state in states
            ],
        },
        "success": {
            "train_all": period_success(success, [int(x) for x in success.index]),
            "train_reference": period_success(success, reference_cycles),
            "train_student": period_success(success, student_cycles),
            "train_student_first20": period_success(
                success, student_cycles[:20]
            ),
            "train_student_last20": period_success(
                success, student_cycles[-20:]
            ),
            "eval_all": period_success(eval_success, eval_cycles),
            "eval_points": [
                {
                    "cycle": int(cycle),
                    "successes": int(round(rate * 4)),
                    "episodes": 4,
                    "rate": float(rate),
                }
                for cycle, rate in eval_success.items()
            ],
        },
        "optimization": {
            tag: metric_stats(values(accumulator, tag))
            for tag in [
                "train/sac/actor_loss",
                "train/sac/critic_loss",
                "train/actor/bc_loss",
                "train/actor/q_value_0",
                "train/actor/q_value_1",
                "train/critic/q_data",
                "train/actor/grad_norm",
                "train/critic/grad_norm",
            ]
        },
        "resources": {
            "samples": int(len(resource)),
            "monitor_duration_seconds": int(
                resource["unix_time"].iloc[-1] - resource["unix_time"].iloc[0]
            ),
            "gpu0_memory_peak_mib": int(resource["gpu0_used_mib"].max()),
            "gpu1_memory_peak_mib": int(resource["gpu1_used_mib"].max()),
            "gpu0_util_active_mean_pct": float(active["gpu0_util_pct"].mean()),
            "gpu1_util_active_mean_pct": float(active["gpu1_util_pct"].mean()),
            "gpu0_util_peak_pct": int(resource["gpu0_util_pct"].max()),
            "gpu1_util_peak_pct": int(resource["gpu1_util_pct"].max()),
            "matched_rss_peak_gib": float(
                resource["matched_total_rss_kib"].max() / 2**20
            ),
            "cgroup_anon_peak_gib": float(
                resource["cgroup_anon_bytes"].max() / 2**30
            ),
            "env_rss_peak_gib": float(resource["env_rss_kib"].max() / 2**20),
            "rollout_rss_peak_gib": float(
                resource["rollout_rss_kib"].max() / 2**20
            ),
            "cgroup_current_peak_gib": float(
                resource["cgroup_current_bytes"].max() / 2**30
            ),
            "cgroup_file_peak_gib": float(
                resource["cgroup_file_bytes"].max() / 2**30
            ),
            "memory_max_gib": 240,
            "cgroup_high_event_delta": int(
                resource["cgroup_high_events"].iloc[-1]
                - resource["cgroup_high_events"].iloc[0]
            ),
            "cgroup_max_event_delta": int(
                resource["cgroup_max_events"].iloc[-1]
                - resource["cgroup_max_events"].iloc[0]
            ),
            "cgroup_oom_event_delta": int(
                resource["cgroup_oom_events"].iloc[-1]
                - resource["cgroup_oom_events"].iloc[0]
            ),
            "cgroup_oom_kill_delta": int(
                resource["cgroup_oom_kill_events"].iloc[-1]
                - resource["cgroup_oom_kill_events"].iloc[0]
            ),
            "host_available_min_gib": float(
                resource["host_available_bytes"].min() / 2**30
            ),
            "disk_available_drop_gib": float(
                (
                    resource["disk_available_bytes"].iloc[0]
                    - resource["disk_available_bytes"].iloc[-1]
                )
                / 2**30
            ),
        },
        "artifacts": {
            "run_root_bytes": 1_541_864_422,
            "checkpoint_count": 10,
            "checkpoint_steps": list(CHECKPOINT_BYTES),
            "final_checkpoint_bytes": CHECKPOINT_BYTES[100],
            "final_checkpoint_trajectory_bytes": CHECKPOINT_TRAJECTORY_BYTES[100],
            "final_checkpoint_trajectory_files": CHECKPOINT_TRAJECTORY_FILES[100],
            "run_pt_file_count": 42_931,
            "media_file_count": 0,
            "all_completion_manifests_complete": all(
                json.loads(path.read_text())["complete"]
                for path in status_dir.glob("rlt_completion_step*.json")
            ),
        },
    }
    status_dir.joinpath("status_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    scalar_long.to_csv(status_dir / "selected_scalars.csv", index=False)

    resource_plot = resource.copy()
    resource_plot["minute"] = (
        resource_plot["unix_time"] - resource_plot["unix_time"].iloc[0]
    ) / 60
    resource_plot["bucket"] = (resource_plot["minute"] / 0.5).astype(int)
    resource_30s = (
        resource_plot.groupby("bucket", as_index=False)
        .agg(
            minute=("minute", "mean"),
            gpu0_used_mib=("gpu0_used_mib", "max"),
            gpu1_used_mib=("gpu1_used_mib", "max"),
            gpu0_util_pct=("gpu0_util_pct", "mean"),
            gpu1_util_pct=("gpu1_util_pct", "mean"),
            cgroup_anon_bytes=("cgroup_anon_bytes", "max"),
            matched_total_rss_kib=("matched_total_rss_kib", "max"),
            env_rss_kib=("env_rss_kib", "max"),
            rollout_rss_kib=("rollout_rss_kib", "max"),
            cgroup_current_bytes=("cgroup_current_bytes", "max"),
            cgroup_file_bytes=("cgroup_file_bytes", "max"),
        )
        .drop(columns="bucket")
    )
    resource_30s.to_csv(status_dir / "resources_30s.csv", index=False)

    visual_data = {
        "cycle": [int(x) for x in success.index],
        "train_success_pct": [round(100 * x, 3) for x in success.values],
        "train_success_roll10_pct": [
            round(100 * x, 3)
            for x in success.rolling(10, min_periods=1).mean().values
        ],
        "eval_cycle": [int(x) for x in eval_success.index],
        "eval_success_pct": [round(100 * x, 3) for x in eval_success.values],
        "critic_loss_cycle": [
            int(x) for x in values(accumulator, "train/sac/critic_loss").index
        ],
        "critic_loss": [
            round(float(x), 7)
            for x in values(accumulator, "train/sac/critic_loss").values
        ],
        "bc_loss": [
            round(float(x), 6)
            for x in values(accumulator, "train/actor/bc_loss").values
        ],
        "actor_grad": [
            round(float(x), 4)
            for x in values(accumulator, "train/actor/grad_norm").values
        ],
        "resource_minute": [
            round(float(x), 2) for x in resource_30s["minute"].values
        ],
        "gpu0_gib": [
            round(float(x) / 1024, 3) for x in resource_30s["gpu0_used_mib"]
        ],
        "gpu1_gib": [
            round(float(x) / 1024, 3) for x in resource_30s["gpu1_used_mib"]
        ],
        "anon_gib": [
            round(float(x) / 2**30, 3)
            for x in resource_30s["cgroup_anon_bytes"]
        ],
        "rss_gib": [
            round(float(x) / 2**20, 3)
            for x in resource_30s["matched_total_rss_kib"]
        ],
        "checkpoint_cycle": list(CHECKPOINT_BYTES),
        "checkpoint_mib": [
            round(CHECKPOINT_BYTES[x] / 2**20, 2) for x in CHECKPOINT_BYTES
        ],
        "checkpoint_replay_mib": [
            round(CHECKPOINT_TRAJECTORY_BYTES[x] / 2**20, 2)
            for x in CHECKPOINT_BYTES
        ],
        "student_switch_cycle": switch_cycle,
        "full_ramp_cycle": ramp_cycle,
    }
    status_dir.joinpath("visual_data.json").write_text(
        json.dumps(visual_data, separators=(",", ":")),
        encoding="utf-8",
    )

    plot_success(
        status_dir / "success_and_schedule.png",
        success,
        eval_success,
        switch_cycle,
        ramp_cycle,
        summary["success"],
    )
    plot_optimization(
        status_dir / "optimization_health.png",
        accumulator,
        switch_cycle,
        ramp_cycle,
    )
    plot_resources(status_dir / "resource_profile.png", resource)
    plot_artifacts(status_dir / "checkpoint_growth.png")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
