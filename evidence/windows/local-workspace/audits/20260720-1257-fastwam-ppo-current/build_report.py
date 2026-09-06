from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
LOG = ROOT / "run_embodiment.log"


def parse_number(block: str, name: str) -> float | None:
    match = re.search(
        rf"(?<![\w/]){re.escape(name)}="
        r"(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan)",
        block,
    )
    if not match or match.group(1).lower() == "nan":
        return None
    return float(match.group(1))


METRIC_KEYS = [
    "episode_len", "num_trajectories", "return", "reward", "success_once",
    "advantages_max", "advantages_mean", "advantages_min", "returns_max",
    "returns_mean", "returns_min", "rewards", "actor/approx_kl",
    "actor/clip_fraction", "actor/clipped_ratio", "actor/entropy_loss",
    "actor/grad_norm", "actor/lr", "actor/policy_loss",
    "actor/policy_loss_abs", "actor/ratio", "actor/ratio_abs",
    "actor/total_loss", "critic/explained_variance", "critic/lr",
    "critic/value_clip_ratio", "critic/value_loss", "actor/run_training",
    "generate_rollouts", "step", "sync_weights", "env/env/bootstrap_step",
    "env/env_interact_step", "rollout/predict",
]


def parse_metrics() -> pd.DataFrame:
    text = LOG.read_text(encoding="utf-8", errors="replace")
    rows: list[dict[str, float | int | None]] = []
    for block in re.split(r"(?=Global Step:)", text):
        match = re.search(r"Global Step:\s*(\d+)/(\d+)", block)
        if not match:
            continue
        row: dict[str, float | int | None] = {
            "global_step": int(match.group(1)),
            "max_steps": int(match.group(2)),
        }
        for key in METRIC_KEYS:
            row[key] = parse_number(block, key)
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("global_step").drop_duplicates(
        "global_step", keep="last"
    )
    if frame.empty:
        raise RuntimeError("No completed metric tables found")
    frame["success_pct"] = frame["success_once"] * 100.0
    frame["success_ma5"] = frame["success_pct"].rolling(5, min_periods=1).mean()
    frame["success_ma10"] = frame["success_pct"].rolling(10, min_periods=3).mean()
    return frame


def success_limits(values: pd.Series) -> tuple[float, float]:
    clean = values.dropna().to_numpy(dtype=float)
    lo = float(np.min(clean))
    hi = float(np.max(clean))
    span = max(hi - lo, 12.0)
    margin = max(3.0, span * 0.18)
    lower = max(0.0, math.floor((lo - margin) / 5.0) * 5.0)
    upper = min(100.0, math.ceil((hi + margin) / 5.0) * 5.0)
    if upper - lower < 20.0:
        midpoint = (upper + lower) / 2.0
        lower = max(0.0, math.floor((midpoint - 10.0) / 5.0) * 5.0)
        upper = min(100.0, lower + 20.0)
    return lower, upper


def step_axis(ax: plt.Axes, xmax: int) -> None:
    ax.set_xlim(0, xmax)
    tick_step = max(1, math.ceil(xmax / 12))
    ax.set_xticks(np.arange(0, xmax + 1, tick_step))
    ax.set_xlabel("Global step (axis starts at 0; data starts at step 1)")


metrics = parse_metrics()
metrics.to_csv(ROOT / "metrics_table.csv", index=False)
resources = pd.read_csv(ROOT / "resources.csv")
resources["timestamp"] = pd.to_datetime(resources["timestamp"])
resources["elapsed_hours"] = (
    resources["timestamp"] - resources["timestamp"].iloc[0]
).dt.total_seconds() / 3600.0
for column in [
    "gpu0_memory_mb", "gpu1_memory_mb", "gpu_total_memory_mb", "env_rss_mb",
    "actor_rss_mb", "rollout_rss_mb",
]:
    resources[column.replace("_mb", "_gib")] = resources[column] / 1024.0

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 220,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.22,
})

steps = metrics["global_step"].to_numpy()
xmax = int(steps.max())
success_lo, success_hi = success_limits(metrics["success_pct"])

# Success is deliberately the dominant, phone-readable chart. Raw values remain
# visible while 5- and 10-step trailing means expose the underlying direction.
fig, ax = plt.subplots(figsize=(11.5, 7.2))
ax.plot(
    steps, metrics["success_pct"], color="#94a3b8", marker="o", markersize=4,
    linewidth=1.25, alpha=0.72, label="Raw success",
)
ax.plot(
    steps, metrics["success_ma5"], color="#2563eb", linewidth=3.0,
    marker="o", markersize=3.2, label="5-step trailing mean",
)
ax.plot(
    steps, metrics["success_ma10"], color="#dc2626", linewidth=2.4,
    label="10-step trailing mean",
)
overall = float(metrics["success_pct"].mean())
ax.axhline(overall, color="#64748b", linestyle="--", linewidth=1.2,
           label=f"Overall mean {overall:.1f}%")
ax.set_ylim(success_lo, success_hi)
ax.set_ylabel("Success rate (%)")
step_axis(ax, xmax)
ax.set_title("Fast-WAM PPO · move_stapler_pad · success rate")
ax.legend(ncol=2, loc="best")
latest = metrics.iloc[-1]
ax.annotate(
    f"step {int(latest['global_step'])}\nraw {latest['success_pct']:.1f}%\nMA5 {latest['success_ma5']:.1f}%",
    xy=(latest["global_step"], latest["success_ma5"]),
    xytext=(-78, 26), textcoords="offset points",
    arrowprops={"arrowstyle": "->", "color": "#2563eb"},
    bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#93c5fd", "alpha": 0.94},
)
fig.tight_layout()
fig.savefig(ROOT / "success_rate_full.png", bbox_inches="tight")
plt.close(fig)

# PPO actor, critic and advantage internals.
fig, axes = plt.subplots(3, 2, figsize=(13.2, 12.2), sharex=True)
axes[0, 0].plot(steps, metrics["actor/policy_loss"], "o-", label="Policy loss")
axes[0, 0].axhline(0, color="#64748b", linewidth=1)
axes[0, 0].set_title("Actor policy loss")

axes[0, 1].plot(steps, metrics["critic/value_loss"], "o-", color="#dc2626", label="Value loss")
axes[0, 1].set_title("Critic value loss")

axes[1, 0].plot(steps, metrics["actor/approx_kl"], "o-", label="Approx KL")
axes[1, 0].plot(steps, metrics["actor/clip_fraction"], "s-", label="Clip fraction")
axes[1, 0].axhline(0, color="#64748b", linewidth=1)
axes[1, 0].set_title("PPO trust-region signals")

axes[1, 1].plot(steps, metrics["actor/ratio_abs"] * 100.0, "o-", label="Mean |ratio-1|")
axes[1, 1].plot(steps, metrics["actor/clip_fraction"] * 100.0, "s-", label="Clip fraction")
axes[1, 1].set_ylabel("Percent")
axes[1, 1].set_title("Policy update magnitude")

axes[2, 0].plot(steps, metrics["actor/grad_norm"], "o-", color="#7c3aed", label="Pre-clip grad norm")
axes[2, 0].axhline(1.0, color="#dc2626", linestyle="--", label="clip_grad=1")
axes[2, 0].set_title("Combined actor + critic gradient")

axes[2, 1].plot(steps, metrics["returns_mean"], "o-", label="Mean GAE return")
axes[2, 1].fill_between(
    steps, metrics["returns_min"], metrics["returns_max"], alpha=0.18,
    label="Return min–max",
)
axes[2, 1].set_title("GAE return target")

for ax in axes.flat:
    step_axis(ax, xmax)
    ax.legend()
fig.suptitle("Fast-WAM PPO · optimization and critic metrics · full history", y=1.005, fontsize=17)
fig.tight_layout()
fig.savefig(ROOT / "ppo_metrics_full.png", bbox_inches="tight")
plt.close(fig)

# Top-level and nested timing.
fig, axes = plt.subplots(2, 1, figsize=(12.5, 9.5), sharex=True)
axes[0].plot(steps, metrics["step"] / 60.0, "o-", linewidth=2.2, label="Step wall time")
axes[0].plot(steps, metrics["generate_rollouts"] / 60.0, "s-", label="Rollout")
axes[0].plot(steps, metrics["actor/run_training"] / 60.0, "^-", label="Actor train")
axes[0].set_ylabel("Minutes")
axes[0].set_title("Top-level time per global step")
axes[0].legend()

axes[1].plot(steps, metrics["env/env/bootstrap_step"] / 60.0, "o-", label="Env bootstrap/reset")
axes[1].plot(steps, metrics["env/env_interact_step"] / 60.0, "s-", label="Env action execution")
axes[1].plot(steps, metrics["rollout/predict"] / 60.0, "^-", label="Fast-WAM predict")
axes[1].set_ylabel("Minutes")
axes[1].set_title("Nested rollout phases (overlap; do not sum)")
axes[1].legend(ncol=2)
for ax in axes:
    step_axis(ax, xmax)
fig.suptitle("Fast-WAM PPO · timing · full history", y=1.005, fontsize=17)
fig.tight_layout()
fig.savefig(ROOT / "timing_full.png", bbox_inches="tight")
plt.close(fig)

# Full-resolution resource history.
hours = resources["elapsed_hours"]
fig, axes = plt.subplots(4, 1, figsize=(13.2, 13.8), sharex=True)
axes[0].plot(hours, resources["cgroup_ram_pct"], linewidth=1.25)
axes[0].axhline(90, color="#dc2626", linestyle="--", linewidth=1.1, label="90%")
axes[0].set_ylim(45, 102)
axes[0].set_ylabel("RAM (%)")
axes[0].set_title("Cgroup host RAM")
axes[0].legend()

axes[1].plot(hours, resources["gpu0_memory_gib"], label="GPU 0", linewidth=1.2)
axes[1].plot(hours, resources["gpu1_memory_gib"], label="GPU 1", linewidth=1.2)
axes[1].set_ylim(0, 80)
axes[1].set_ylabel("GiB")
axes[1].set_title("A800 memory")
axes[1].legend()

axes[2].plot(hours, resources["gpu0_util_pct"], label="GPU 0", linewidth=0.8)
axes[2].plot(hours, resources["gpu1_util_pct"], label="GPU 1", linewidth=0.8)
axes[2].set_ylim(0, 102)
axes[2].set_ylabel("Utilization (%)")
axes[2].set_title("Instantaneous GPU utilization")
axes[2].legend()

axes[3].plot(hours, resources["env_rss_gib"], label="Environment", linewidth=1.05)
axes[3].plot(hours, resources["actor_rss_gib"], label="Actor", linewidth=1.05)
axes[3].plot(hours, resources["rollout_rss_gib"], label="Rollout", linewidth=1.05)
axes[3].set_ylabel("RSS (GiB)")
axes[3].set_xlabel("Elapsed wall time (hours from monitor start)")
axes[3].set_title("Worker process RSS")
axes[3].legend(ncol=3)
for ax in axes:
    ax.set_xlim(0, float(hours.iloc[-1]))
fig.suptitle("Fast-WAM PPO · resources · full run history", y=1.0, fontsize=17)
fig.tight_layout()
fig.savefig(ROOT / "resources_full.png", bbox_inches="tight")
plt.close(fig)

summary = {
    "snapshot_time": str(resources["timestamp"].iloc[-1]),
    "completed_steps": int(metrics["global_step"].max()),
    "latest_success_pct": float(metrics["success_pct"].iloc[-1]),
    "success_ma5_pct": float(metrics["success_ma5"].iloc[-1]),
    "success_ma10_pct": float(metrics["success_ma10"].iloc[-1]),
    "mean_success_pct": float(metrics["success_pct"].mean()),
    "best_success_pct": float(metrics["success_pct"].max()),
    "success_axis": [success_lo, success_hi],
    "mean_step_minutes": float(metrics["step"].mean() / 60.0),
    "mean_rollout_minutes": float(metrics["generate_rollouts"].mean() / 60.0),
    "mean_actor_minutes": float(metrics["actor/run_training"].mean() / 60.0),
    "latest_policy_loss": float(metrics["actor/policy_loss"].iloc[-1]),
    "latest_value_loss": float(metrics["critic/value_loss"].iloc[-1]),
    "latest_approx_kl": float(metrics["actor/approx_kl"].iloc[-1]),
    "latest_clip_fraction": float(metrics["actor/clip_fraction"].iloc[-1]),
    "latest_ratio_abs": float(metrics["actor/ratio_abs"].iloc[-1]),
    "latest_grad_norm": float(metrics["actor/grad_norm"].iloc[-1]),
    "ram_current_pct": float(resources["cgroup_ram_pct"].iloc[-1]),
    "ram_peak_pct": float(resources["cgroup_ram_pct"].max()),
    "gpu0_current_gib": float(resources["gpu0_memory_gib"].iloc[-1]),
    "gpu1_current_gib": float(resources["gpu1_memory_gib"].iloc[-1]),
    "gpu0_peak_gib": float(resources["gpu0_memory_gib"].max()),
    "gpu1_peak_gib": float(resources["gpu1_memory_gib"].max()),
    "cgroup_oom": int(resources["cgroup_oom"].max()),
    "cgroup_oom_kill": int(resources["cgroup_oom_kill"].max()),
}
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

interactive_columns = [
    "global_step", "success_pct", "success_ma5", "success_ma10",
    "actor/policy_loss", "critic/value_loss", "actor/approx_kl",
    "actor/clip_fraction", "actor/ratio_abs", "actor/grad_norm", "step",
    "generate_rollouts", "actor/run_training",
]
(ROOT / "interactive_data.json").write_text(
    metrics[interactive_columns].to_json(orient="records"), encoding="utf-8"
)
print(json.dumps(summary, indent=2))
