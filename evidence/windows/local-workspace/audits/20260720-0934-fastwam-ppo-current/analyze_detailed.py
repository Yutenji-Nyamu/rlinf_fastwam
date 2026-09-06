from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def number(block: str, name: str):
    match = re.search(
        rf"(?<![\w/]){re.escape(name)}="
        r"(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan)",
        block,
    )
    if not match or match.group(1).lower() == "nan":
        return None
    return float(match.group(1))


metric_keys = [
    "episode_len", "num_trajectories", "return", "reward", "success_once",
    "advantages_max", "advantages_mean", "advantages_min",
    "returns_max", "returns_mean", "returns_min", "rewards",
    "actor/approx_kl", "actor/clip_fraction", "actor/clipped_ratio",
    "actor/dual_cliped_ratio", "actor/entropy_loss", "actor/grad_norm",
    "actor/lr", "actor/policy_loss", "actor/policy_loss_abs",
    "actor/ratio", "actor/ratio_abs", "actor/total_loss",
    "critic/explained_variance", "critic/lr", "critic/value_clip_ratio",
    "critic/value_loss",
]

time_keys = [
    "actor/actor/compute_adv", "actor/actor/recv_traj",
    "actor/actor/sync_model_to_rollout", "actor/run_training",
    "cal_adv_and_returns", "env/compute_bootstrap_rewards",
    "env/env/bootstrap_step", "env/env/send_rollout_trajectories",
    "env/env_interact_step", "env/interact", "env/run_interact_once",
    "generate_rollouts", "rollout/generate_one_epoch", "rollout/predict",
    "rollout/rollout/generate", "step", "sync_weights",
]

text = (ROOT / "metrics.log").read_text(encoding="utf-8", errors="replace")
rows = []
for block in re.split(r"(?=Global Step:)", text):
    match = re.search(r"Global Step:\s*(\d+)/(\d+)", block)
    if not match:
        continue
    row = {"global_step": int(match.group(1)), "max_steps": int(match.group(2))}
    for key in metric_keys + time_keys:
        row[key] = number(block, key)
    rows.append(row)

metrics = pd.DataFrame(rows).sort_values("global_step")
if metrics.empty:
    raise RuntimeError("No completed metric tables")
metrics.to_csv(ROOT / "metrics_table.csv", index=False)

resources = pd.read_csv(ROOT / "resources.csv")
resources["timestamp"] = pd.to_datetime(resources["timestamp"])
resources["elapsed_hours"] = (
    resources["timestamp"] - resources["timestamp"].iloc[0]
).dt.total_seconds() / 3600
for col in ["gpu0_memory_mb", "gpu1_memory_mb", "gpu_total_memory_mb"]:
    resources[col.replace("_mb", "_gib")] = resources[col] / 1024

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 180, "font.size": 11,
    "axes.titlesize": 14, "axes.labelsize": 11, "legend.fontsize": 9,
    "axes.grid": True, "grid.alpha": 0.22,
})
steps = metrics["global_step"].to_numpy()
xmax = max(1, int(steps.max()))
xticks = np.arange(0, xmax + 1, 1)


def step_axis(ax):
    ax.set_xlim(0, xmax)
    ax.set_xticks(xticks)
    ax.set_xlabel("Global step (axis starts at 0; no synthetic step-0 value)")


# Environment and rollout metrics.
fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
axes[0, 0].plot(steps, metrics["success_once"] * 100, "o-", lw=2.5, label="Success once")
axes[0, 0].plot(steps, metrics["return"] * 100, "s--", lw=1.5, label="Episode return")
axes[0, 0].set_ylim(0, 100)
axes[0, 0].set_ylabel("Percent")
axes[0, 0].set_title("Environment success and return")
axes[0, 0].legend()

axes[0, 1].plot(steps, metrics["returns_mean"], "o-", lw=2, label="Mean")
axes[0, 1].fill_between(steps, metrics["returns_min"], metrics["returns_max"], alpha=.18, label="Min–max")
axes[0, 1].set_ylabel("GAE return")
axes[0, 1].set_title("PPO returns")
axes[0, 1].legend()

axes[1, 0].plot(steps, metrics["advantages_max"], "o-", label="Max")
axes[1, 0].plot(steps, metrics["advantages_min"], "s-", label="Min")
axes[1, 0].plot(steps, metrics["advantages_mean"], "^-", label="Mean")
axes[1, 0].axhline(0, color="#64748b", lw=1)
axes[1, 0].set_ylabel("Normalized advantage")
axes[1, 0].set_title("Advantage range")
axes[1, 0].legend()

axes[1, 1].plot(steps, metrics["reward"], "o-", label="Environment reward")
axes[1, 1].plot(steps, metrics["rewards"], "s--", label="Rollout reward")
axes[1, 1].set_ylabel("Mean reward per step")
axes[1, 1].set_title("Reward signals")
axes[1, 1].legend()
for ax in axes.flat:
    step_axis(ax)
fig.suptitle("Fast-WAM PPO · environment and rollout metrics · full completed history", y=1.01, fontsize=17)
fig.tight_layout()
fig.savefig(ROOT / "rl_environment_rollout_full.png", bbox_inches="tight")
plt.close(fig)

# Actor and critic metrics.
fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharex=True)
axes[0, 0].plot(steps, metrics["actor/policy_loss"], "o-", label="Policy loss")
axes[0, 0].plot(steps, metrics["actor/total_loss"], "s--", label="Total loss")
axes[0, 0].axhline(0, color="#64748b", lw=1)
axes[0, 0].set_title("Actor losses")
axes[0, 0].legend()

axes[0, 1].plot(steps, metrics["actor/ratio"], "o-", label="Ratio")
axes[0, 1].plot(steps, metrics["actor/clipped_ratio"], "s--", label="Clipped ratio")
axes[0, 1].axhline(1, color="#64748b", lw=1)
axes[0, 1].set_title("PPO ratios")
axes[0, 1].legend()

axes[1, 0].plot(steps, metrics["actor/approx_kl"], "o-", label="Approx KL")
axes[1, 0].plot(steps, metrics["actor/clip_fraction"], "s-", label="Clip fraction")
axes[1, 0].axhline(0, color="#64748b", lw=1)
axes[1, 0].set_title("Trust-region signals")
axes[1, 0].legend()

axes[1, 1].plot(steps, metrics["actor/grad_norm"], "o-", label="Pre-clip grad norm")
axes[1, 1].set_title("Gradient norm (clip_grad=1.0)")
axes[1, 1].legend()

axes[2, 0].plot(steps, metrics["critic/value_loss"], "o-", label="Value loss")
axes[2, 0].plot(steps, metrics["critic/value_clip_ratio"], "s--", label="Value clip ratio")
axes[2, 0].set_title("Critic metrics")
axes[2, 0].legend()

axes[2, 1].axis("off")
axes[2, 1].text(.02, .76, "Constant / unavailable metrics", fontsize=14)
axes[2, 1].text(.02, .57, "Actor LR: 5.0e-6")
axes[2, 1].text(.02, .43, "Critic LR: 1.1e-4")
axes[2, 1].text(.02, .29, "Entropy loss: 0.0 (entropy bonus disabled)")
axes[2, 1].text(.02, .15, f"Explained variance: NaN in {len(metrics)}/{len(metrics)} steps")
for ax in axes.flat:
    if ax.axison:
        step_axis(ax)
fig.suptitle("Fast-WAM PPO · actor and critic metrics · full completed history", y=1.0, fontsize=17)
fig.tight_layout()
fig.savefig(ROOT / "rl_actor_critic_full.png", bbox_inches="tight")
plt.close(fig)

# Timing: top-level timers are sequential; nested timers overlap and are shown separately.
top = pd.DataFrame({
    "Rollout": metrics["generate_rollouts"],
    "Actor train": metrics["actor/run_training"],
    "Weight sync": metrics["sync_weights"],
}) / 60
top["Other"] = (metrics["step"] / 60 - top.sum(axis=1)).clip(lower=0)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0, 0].plot(steps, metrics["step"] / 60, "o-", lw=2.5)
axes[0, 0].set_ylabel("Minutes")
axes[0, 0].set_title("Wall time per global step")
step_axis(axes[0, 0])

bottom = np.zeros(len(metrics))
for label, color in zip(top.columns, ["#2563eb", "#dc2626", "#16a34a", "#94a3b8"]):
    vals = top[label].to_numpy()
    axes[0, 1].bar(steps, vals, bottom=bottom, label=label, color=color)
    bottom += vals
axes[0, 1].set_ylabel("Minutes")
axes[0, 1].set_title("Top-level sequential time decomposition")
axes[0, 1].legend()
step_axis(axes[0, 1])

nested = pd.Series({
    "Env interact": metrics["env/run_interact_once"].mean(),
    "Env action execution": metrics["env/env_interact_step"].mean(),
    "Env bootstrap": metrics["env/env/bootstrap_step"].mean(),
    "Rollout worker epoch": metrics["rollout/generate_one_epoch"].mean(),
    "Fast-WAM predict": metrics["rollout/predict"].mean(),
    "Actor training": metrics["actor/run_training"].mean(),
    "Weight sync": metrics["sync_weights"].mean(),
}).sort_values()
axes[1, 0].barh(nested.index, nested.values / 60, color="#0f766e")
axes[1, 0].set_xlabel("Mean minutes per global step")
axes[1, 0].set_title("Nested timers (overlap; do not sum)")

axes[1, 1].axis("off")
avg_step = metrics["step"].mean()
avg_rollout = metrics["generate_rollouts"].mean()
avg_actor = metrics["actor/run_training"].mean()
axes[1, 1].text(.02, .80, "Five-step timing summary", fontsize=14)
axes[1, 1].text(.02, .63, f"Mean step: {avg_step/60:.2f} min")
axes[1, 1].text(.02, .49, f"Mean rollout: {avg_rollout/60:.2f} min ({avg_rollout/avg_step*100:.1f}%)")
axes[1, 1].text(.02, .35, f"Mean actor train: {avg_actor/60:.2f} min ({avg_actor/avg_step*100:.1f}%)")
axes[1, 1].text(.02, .21, f"Mean weight sync: {metrics['sync_weights'].mean():.2f} s")
fig.suptitle("Fast-WAM PPO · timing · full completed history", y=1.0, fontsize=17)
fig.tight_layout()
fig.savefig(ROOT / "time_breakdown_full.png", bbox_inches="tight")
plt.close(fig)

# Resources: full-resolution timeline.
fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
axes[0].plot(resources["elapsed_hours"], resources["cgroup_ram_pct"], lw=1.4)
axes[0].axhline(90, color="#dc2626", ls="--", lw=1.2, label="90%")
axes[0].set_ylim(0, 102)
axes[0].set_ylabel("RAM (%)")
axes[0].set_title("Cgroup host RAM")
axes[0].legend()
axes[1].plot(resources["elapsed_hours"], resources["gpu0_memory_gib"], label="GPU 0", lw=1.3)
axes[1].plot(resources["elapsed_hours"], resources["gpu1_memory_gib"], label="GPU 1", lw=1.3)
axes[1].set_ylim(0, 80)
axes[1].set_ylabel("Memory (GiB)")
axes[1].set_title("A800 memory")
axes[1].legend()
axes[2].plot(resources["elapsed_hours"], resources["gpu0_util_pct"], label="GPU 0", lw=.9)
axes[2].plot(resources["elapsed_hours"], resources["gpu1_util_pct"], label="GPU 1", lw=.9)
axes[2].set_ylim(0, 102)
axes[2].set_ylabel("Utilization (%)")
axes[2].set_xlabel("Elapsed wall time (hours from monitor start)")
axes[2].set_title("Instantaneous GPU utilization")
axes[2].legend()
for ax in axes:
    ax.set_xlim(0, resources["elapsed_hours"].iloc[-1])
fig.suptitle("Fast-WAM PPO · resources · full run history", y=1.0, fontsize=17)
fig.tight_layout()
fig.savefig(ROOT / "resources_detailed_full.png", bbox_inches="tight")
plt.close(fig)

# Compact inline data, preserving all scalar metrics and downsampling resource time series.
target = 260
idx = np.linspace(0, len(resources) - 1, min(target, len(resources)), dtype=int)
resource_view = resources.iloc[idx]
after_10m = resources[resources["elapsed_hours"] >= (10 / 60)]
active = resources[resources["gpu_total_memory_gib"] >= 20]

def records(frame):
    return json.loads(frame.to_json(orient="records"))

summary = {
    "metrics": records(metrics),
    "resources": records(resource_view[[
        "elapsed_hours", "cgroup_ram_pct", "gpu0_memory_gib", "gpu1_memory_gib",
        "gpu0_util_pct", "gpu1_util_pct", "gpu0_power_w", "gpu1_power_w",
        "env_rss_mb", "actor_rss_mb", "rollout_rss_mb",
    ]]),
    "stats": {
        "completed_steps": int(metrics["global_step"].max()),
        "mean_step_seconds": float(metrics["step"].mean()),
        "mean_rollout_seconds": float(metrics["generate_rollouts"].mean()),
        "mean_actor_seconds": float(metrics["actor/run_training"].mean()),
        "mean_sync_seconds": float(metrics["sync_weights"].mean()),
        "latest_success": float(metrics["success_once"].iloc[-1]),
        "best_success": float(metrics["success_once"].max()),
        "mean_success": float(metrics["success_once"].mean()),
        "ram_current_pct": float(resources["cgroup_ram_pct"].iloc[-1]),
        "ram_peak_pct": float(resources["cgroup_ram_pct"].max()),
        "ram_peak_after_10m_pct": float(after_10m["cgroup_ram_pct"].max()),
        "ram_p95_pct": float(resources["cgroup_ram_pct"].quantile(.95)),
        "ram_over_90_fraction": float((resources["cgroup_ram_pct"] >= 90).mean()),
        "gpu0_current_gib": float(resources["gpu0_memory_gib"].iloc[-1]),
        "gpu1_current_gib": float(resources["gpu1_memory_gib"].iloc[-1]),
        "gpu0_peak_gib": float(resources["gpu0_memory_gib"].max()),
        "gpu1_peak_gib": float(resources["gpu1_memory_gib"].max()),
        "gpu_total_active_mean_gib": float(active["gpu_total_memory_gib"].mean()),
        "gpu_active_median_imbalance_gib": float((active["gpu0_memory_gib"] - active["gpu1_memory_gib"]).abs().median()),
        "gpu0_util_mean_pct": float(active["gpu0_util_pct"].mean()),
        "gpu1_util_mean_pct": float(active["gpu1_util_pct"].mean()),
        "cgroup_oom": int(resources["cgroup_oom"].max()),
        "cgroup_oom_kill": int(resources["cgroup_oom_kill"].max()),
        "resource_hours": float(resources["elapsed_hours"].iloc[-1]),
    },
}
(ROOT / "analysis_detailed.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary["stats"], indent=2))
