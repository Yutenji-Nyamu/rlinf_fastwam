"""Analyze a partial RoboTwin RLT Stage 2 formal250 snapshot."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


GIB = 1024**3
KIB_TO_GIB = 1 / 1024**2


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_accumulator(evidence: Path) -> EventAccumulator:
    events = sorted(evidence.glob("events.out.tfevents.*"))
    if len(events) != 1:
        raise RuntimeError(f"expected one event file, got {events}")
    accumulator = EventAccumulator(
        str(events[0]),
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()
    return accumulator


def series(accumulator: EventAccumulator, tag: str) -> pd.Series:
    if tag not in accumulator.Tags().get("scalars", []):
        return pd.Series(dtype=float)
    return pd.Series(
        {int(event.step) + 1: float(event.value) for event in accumulator.Scalars(tag)},
        dtype=float,
    ).sort_index()


def finite_stats(values: pd.Series) -> dict[str, float | int | None]:
    clean = values[np.isfinite(values)]
    if clean.empty:
        return {
            "count": 0,
            "first": None,
            "last": None,
            "min": None,
            "max": None,
            "mean": None,
            "last10_mean": None,
        }
    return {
        "count": int(clean.size),
        "first": float(clean.iloc[0]),
        "last": float(clean.iloc[-1]),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "mean": float(clean.mean()),
        "last10_mean": float(clean.iloc[-10:].mean()),
    }


def weighted_success(
    success_rate: pd.Series,
    denominators: pd.Series,
) -> dict[str, float | int]:
    aligned = pd.concat(
        [success_rate.rename("rate"), denominators.rename("denominator")],
        axis=1,
        join="inner",
    ).dropna()
    episodes = int(round(aligned["denominator"].sum()))
    successes = int(round((aligned["rate"] * aligned["denominator"]).sum()))
    return {
        "cycles": int(aligned.shape[0]),
        "episodes": episodes,
        "successes": successes,
        "rate": successes / episodes if episodes else 0.0,
    }


def projected_cycle(current_cycle: int, current_value: float, target: float) -> int | None:
    if current_cycle <= 0 or current_value <= 0:
        return None
    return int(math.ceil(target / (current_value / current_cycle)))


def parse_latest_console_status(driver_log: str) -> dict[str, str | int | float | None]:
    matches = list(
        re.finditer(
            r"Global Step:\s+(\d+)/(\d+).*?"
            r"Elapsed:\s+([0-9:]+)\s+│\s+ETA:\s+([0-9:]+).*?"
            r"Step Time:\s+([0-9.]+)s",
            driver_log,
            flags=re.DOTALL,
        )
    )
    if not matches:
        return {
            "cycle": None,
            "total": None,
            "elapsed": None,
            "eta": None,
            "step_seconds": None,
        }
    match = matches[-1]
    return {
        "cycle": int(match.group(1)),
        "total": int(match.group(2)),
        "elapsed": match.group(3),
        "eta": match.group(4),
        "step_seconds": float(match.group(5)),
    }


def resource_stats(resource: pd.DataFrame) -> dict:
    timestamp = pd.to_datetime(resource["unix_time"], unit="s", utc=True)
    positive_gaps = resource["unix_time"].diff()
    positive_gaps = positive_gaps[positive_gaps > 0]
    active = resource["compute_process_count"] > 0

    def bytes_stats(column: str) -> dict[str, float]:
        values = resource[column] / GIB
        return {
            "current_gib": float(values.iloc[-1]),
            "min_gib": float(values.min()),
            "max_gib": float(values.max()),
        }

    def kib_stats(column: str) -> dict[str, float]:
        values = resource[column] * KIB_TO_GIB
        return {
            "current_gib": float(values.iloc[-1]),
            "min_gib": float(values.min()),
            "max_gib": float(values.max()),
        }

    def event_delta(column: str) -> int:
        return int(resource[column].iloc[-1] - resource[column].iloc[0])

    return {
        "rows": int(len(resource)),
        "snapshot_time": timestamp.iloc[-1]
        .tz_convert(ZoneInfo("Asia/Shanghai"))
        .isoformat(),
        "coverage_seconds": float(
            resource["unix_time"].iloc[-1] - resource["unix_time"].iloc[0]
        ),
        "sample_interval_median_seconds": (
            float(positive_gaps.median()) if not positive_gaps.empty else None
        ),
        "sample_interval_max_seconds": (
            float(positive_gaps.max()) if not positive_gaps.empty else None
        ),
        "gpu0_peak_mib": int(resource["gpu0_used_mib"].max()),
        "gpu1_peak_mib": int(resource["gpu1_used_mib"].max()),
        "gpu0_current_mib": int(resource["gpu0_used_mib"].iloc[-1]),
        "gpu1_current_mib": int(resource["gpu1_used_mib"].iloc[-1]),
        "gpu0_active_mean_util_pct": float(resource.loc[active, "gpu0_util_pct"].mean()),
        "gpu1_active_mean_util_pct": float(resource.loc[active, "gpu1_util_pct"].mean()),
        "gpu0_active_p95_util_pct": float(
            resource.loc[active, "gpu0_util_pct"].quantile(0.95)
        ),
        "gpu1_active_p95_util_pct": float(
            resource.loc[active, "gpu1_util_pct"].quantile(0.95)
        ),
        "host_available": bytes_stats("host_available_bytes"),
        "disk_available": bytes_stats("disk_available_bytes"),
        "cgroup_current": bytes_stats("cgroup_current_bytes"),
        "cgroup_anon": bytes_stats("cgroup_anon_bytes"),
        "cgroup_file": bytes_stats("cgroup_file_bytes"),
        "env_rss": kib_stats("env_rss_kib"),
        "rollout_rss": kib_stats("rollout_rss_kib"),
        "matched_total_rss": kib_stats("matched_total_rss_kib"),
        "high_event_delta": event_delta("cgroup_high_events"),
        "max_event_delta": event_delta("cgroup_max_events"),
        "oom_event_delta": event_delta("cgroup_oom_events"),
        "oom_kill_event_delta": event_delta("cgroup_oom_kill_events"),
    }


def plot_success(
    output: Path,
    train_success: pd.Series,
    eval_success: pd.Series,
    latest_cycle: int,
    warmup_projection: int | None,
    summary: dict,
) -> None:
    configure_plotting()
    rolling = train_success.rolling(10, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.scatter(
        train_success.index,
        train_success.values * 100,
        s=22,
        alpha=0.45,
        label="Train success / cycle (8 episodes)",
    )
    ax.plot(
        rolling.index,
        rolling.values * 100,
        linewidth=2.4,
        label="Train rolling 10-cycle (80 episodes)",
    )
    if not eval_success.empty:
        ax.plot(
            eval_success.index,
            eval_success.values * 100,
            marker="D",
            markersize=7,
            color="tab:red",
            linewidth=1.5,
            label="Deterministic student eval (20 episodes)",
        )
    ax.axvspan(1, latest_cycle + 0.5, color="tab:gray", alpha=0.08)
    ax.axvline(latest_cycle, color="black", linewidth=1.2, label="Current snapshot")
    view_end = max(50, latest_cycle + 5)
    if warmup_projection is not None and warmup_projection <= view_end:
        ax.axvline(
            warmup_projection,
            color="tab:purple",
            linestyle="--",
            linewidth=1.4,
            label=f"Projected replay gate ~{warmup_projection}",
        )
    train = summary["train_success"]
    last10 = summary["train_last10_success"]
    eval_total = summary["eval_success"]
    ax.text(
        0.02,
        0.97,
        (
            "Current controller: frozen reference (no RL updates yet)\n"
            f"Train: {train['successes']}/{train['episodes']} "
            f"({100 * train['rate']:.1f}%)\n"
            f"Last 10 cycles: {last10['successes']}/{last10['episodes']} "
            f"({100 * last10['rate']:.1f}%)\n"
            f"Eval: {eval_total['successes']}/{eval_total['episodes']} "
            f"({100 * eval_total['rate']:.1f}%)\n"
            f"Projected replay gate: ~cycle {warmup_projection}"
        ),
        transform=ax.transAxes,
        va="top",
    )
    ax.set(
        title="RLT formal250: success during reference collection",
        xlabel="Outer cycle",
        ylabel="Success rate (%)",
        xlim=(1, view_end),
        ylim=(-2, max(45, float(train_success.max() * 100 + 8))),
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_replay_timing(
    output: Path,
    min_replay: pd.Series,
    global_transitions: pd.Series,
    step_time: pd.Series,
    rollout_time: pd.Series,
    eval_time: pd.Series,
    latest_cycle: int,
) -> None:
    configure_plotting()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    axes[0].plot(
        min_replay.index,
        min_replay.values,
        linewidth=2.2,
        label="Minimum replay size / rank",
    )
    axes[0].plot(
        global_transitions.index,
        global_transitions.values,
        linewidth=2,
        label="Global macro transitions",
    )
    axes[0].axhline(10_000, color="tab:red", linestyle="--", label="Replay gate 10k/rank")
    axes[0].set(
        title="Replay accumulation before the first optimizer update",
        xlabel="Outer cycle",
        ylabel="Rows / transitions",
        xlim=(1, max(40, latest_cycle + 2)),
    )
    axes[0].legend()

    axes[1].plot(step_time.index, step_time.values, label="Total cycle time")
    axes[1].plot(rollout_time.index, rollout_time.values, label="Rollout time")
    for cycle, value in eval_time.items():
        axes[1].scatter(
            [cycle],
            [step_time.get(cycle, value)],
            marker="D",
            s=55,
            color="tab:red",
            label="Eval + checkpoint cycle" if cycle == eval_time.index[0] else None,
        )
    axes[1].set(
        title="Cycle time; red diamonds mark 20-episode eval + checkpoint",
        xlabel="Outer cycle",
        ylabel="Seconds",
        xlim=(1, max(40, latest_cycle + 2)),
    )
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_resources(output: Path, resource: pd.DataFrame, stats: dict) -> None:
    configure_plotting()
    frame = resource.copy()
    frame["minute"] = (frame["unix_time"] - frame["unix_time"].iloc[0]) / 60
    window = max(1, int(round(60 / max(1, resource["unix_time"].diff().median()))))
    frame["gpu0_util_roll"] = frame["gpu0_util_pct"].rolling(
        window, min_periods=1
    ).mean()
    frame["gpu1_util_roll"] = frame["gpu1_util_pct"].rolling(
        window, min_periods=1
    ).mean()

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(frame["minute"], frame["gpu0_used_mib"] / 1024, label="GPU0 memory")
    axes[0].plot(frame["minute"], frame["gpu1_used_mib"] / 1024, label="GPU1 memory")
    axes[0].set_ylabel("GPU memory (GiB)")
    utilization = axes[0].twinx()
    utilization.plot(
        frame["minute"],
        frame["gpu0_util_roll"],
        alpha=0.55,
        linewidth=1,
        label="GPU0 util, 60s mean",
    )
    utilization.plot(
        frame["minute"],
        frame["gpu1_util_roll"],
        alpha=0.55,
        linewidth=1,
        label="GPU1 util, 60s mean",
    )
    utilization.set_ylabel("GPU utilization (%)")
    axes[0].legend(loc="upper left", ncol=2)
    utilization.legend(loc="upper right", ncol=2)

    axes[1].plot(
        frame["minute"],
        frame["cgroup_current_bytes"] / GIB,
        label="cgroup current",
    )
    axes[1].plot(
        frame["minute"],
        frame["cgroup_file_bytes"] / GIB,
        label="file cache",
    )
    axes[1].plot(
        frame["minute"],
        frame["cgroup_anon_bytes"] / GIB,
        label="anonymous",
    )
    axes[1].plot(
        frame["minute"],
        frame["matched_total_rss_kib"] * KIB_TO_GIB,
        label="matched RSS",
    )
    axes[1].axhline(240, color="tab:red", linestyle="--", label="cgroup limit")
    axes[1].set_ylabel("Memory (GiB)")
    axes[1].legend(ncol=3)

    axes[2].plot(
        frame["minute"],
        frame["env_rss_kib"] * KIB_TO_GIB,
        label="Env RSS",
    )
    axes[2].plot(
        frame["minute"],
        frame["rollout_rss_kib"] * KIB_TO_GIB,
        label="Rollout RSS",
    )
    axes[2].plot(
        frame["minute"],
        frame["matched_total_rss_kib"] * KIB_TO_GIB,
        label="Matched total RSS",
    )
    axes[2].set(xlabel="Minutes since monitor start", ylabel="Process RSS (GiB)")
    axes[2].legend(ncol=3)
    axes[2].text(
        0.01,
        0.02,
        (
            f"OOM/OOM-kill delta: {stats['oom_event_delta']}/"
            f"{stats['oom_kill_event_delta']} | "
            f"high/max delta: {stats['high_event_delta']}/"
            f"{stats['max_event_delta']}"
        ),
        transform=axes[2].transAxes,
        va="bottom",
    )
    axes[0].set_title("RLT formal250 resource profile")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_dir")
    args = parser.parse_args()

    evidence = Path(args.evidence_dir)
    accumulator = load_accumulator(evidence)
    resource = pd.read_csv(evidence / "resources.csv")
    driver_log = (evidence / "driver.log").read_text(encoding="utf-8", errors="replace")

    train_denominator = series(accumulator, "env/num_trajectories")
    train_success = series(accumulator, "env/success_once")
    eval_denominator = series(accumulator, "eval/num_trajectories")
    eval_success = series(accumulator, "eval/success_once")
    latest_cycle = int(train_success.index.max())

    train_total = weighted_success(train_success, train_denominator)
    last10_cycles = train_success.index[-10:]
    train_last10 = weighted_success(
        train_success.loc[last10_cycles],
        train_denominator.loc[last10_cycles],
    )
    eval_total = weighted_success(eval_success, eval_denominator)

    min_replay = series(accumulator, "train/rlt/global_min_replay_size")
    global_transitions = series(
        accumulator,
        "train/rlt/global_total_transitions_added",
    )
    update_step = series(accumulator, "train/rlt/update_step")
    critic_updates_run = series(accumulator, "train/rlt/critic_updates_run")
    actor_updates_run = series(accumulator, "train/rlt/actor_updates_run")
    actor_switch = series(accumulator, "train/replay/actor_switch_rate")
    step_time = series(accumulator, "time/step")
    rollout_time = series(accumulator, "time/generate_rollouts")
    eval_time = series(accumulator, "time/eval")

    replay_projection = projected_cycle(
        latest_cycle,
        float(min_replay.iloc[-1]),
        10_000,
    )
    resource_summary = resource_stats(resource)
    duration_hours = resource_summary["coverage_seconds"] / 3600
    checkpoint_completions = sorted(evidence.glob("rlt_completion_step*.json"))
    checkpoint_summary = []
    for path in checkpoint_completions:
        payload = json.loads(path.read_text(encoding="utf-8"))
        step = int(re.search(r"step(\d+)", path.name).group(1))
        replay = {}
        for rank in (0, 1):
            replay_path = evidence / f"replay_rank_{rank}_metadata_step{step}.json"
            replay[str(rank)] = json.loads(replay_path.read_text(encoding="utf-8"))
        checkpoint_summary.append(
            {
                "step": step,
                "complete": bool(payload["complete"]),
                "actor_world_size": int(payload["actor_world_size"]),
                "update_step": int(payload["update_step"]),
                "contract_sha256": payload["rlt_resume_contract_sha256"],
                "replay": replay,
            }
        )

    available_tags = set(accumulator.Tags().get("scalars", []))
    optimization_tags = sorted(
        tag
        for tag in available_tags
        if tag.startswith(("train/sac/", "train/actor/", "train/critic/"))
    )
    summary = {
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "snapshot_time": resource_summary["snapshot_time"],
        "latest_complete_cycle": latest_cycle,
        "progress_fraction": latest_cycle / 250,
        "train_episodes": int(train_denominator.sum()),
        "eval_episodes": int(eval_denominator.sum()) if not eval_denominator.empty else 0,
        "train_success": train_total,
        "train_last10_success": train_last10,
        "eval_success": eval_total,
        "eval_points": [
            {
                "cycle": int(cycle),
                "episodes": int(eval_denominator.loc[cycle]),
                "successes": int(
                    round(eval_success.loc[cycle] * eval_denominator.loc[cycle])
                ),
                "rate": float(eval_success.loc[cycle]),
            }
            for cycle in eval_success.index
        ],
        "global_macro_transitions": int(round(global_transitions.iloc[-1])),
        "macro_transitions_per_cycle": float(global_transitions.iloc[-1] / latest_cycle),
        "minimum_replay_per_rank": int(round(min_replay.iloc[-1])),
        "replay_gate_progress": float(min_replay.iloc[-1] / 10_000),
        "projected_replay_gate_cycle": replay_projection,
        "critic_updates_total": int(round(critic_updates_run.sum())),
        "actor_updates_total": int(round(actor_updates_run.sum())),
        "latest_update_step_pre_cycle": int(round(update_step.iloc[-1])),
        "actor_switch_rate_latest": float(actor_switch.iloc[-1]),
        "optimization_tags_present": optimization_tags,
        "console_status": parse_latest_console_status(driver_log),
        "time_step": finite_stats(step_time),
        "time_rollout": finite_stats(rollout_time),
        "time_eval": finite_stats(eval_time),
        "throughput": {
            "train_episodes_per_hour": (
                float(train_denominator.sum() / duration_hours)
                if duration_hours > 0
                else None
            ),
            "macro_transitions_per_hour": (
                float(global_transitions.iloc[-1] / duration_hours)
                if duration_hours > 0
                else None
            ),
        },
        "resources": resource_summary,
        "checkpoints": checkpoint_summary,
    }
    (evidence / "status_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    configure_plotting()
    plot_success(
        evidence / "success_and_phase.png",
        train_success,
        eval_success,
        latest_cycle,
        replay_projection,
        summary,
    )
    plot_replay_timing(
        evidence / "replay_and_timing.png",
        min_replay,
        global_transitions,
        step_time,
        rollout_time,
        eval_time,
        latest_cycle,
    )
    plot_resources(
        evidence / "resource_profile.png",
        resource,
        resource_summary,
    )
    print(
        json.dumps(
            {
                "summary": str(evidence / "status_summary.json"),
                "latest_cycle": latest_cycle,
                "plots": [
                    str(evidence / "success_and_phase.png"),
                    str(evidence / "replay_and_timing.png"),
                    str(evidence / "resource_profile.png"),
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
