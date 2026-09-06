from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def parse_number(block: str, name: str):
    match = re.search(
        rf"(?<![\w/]){re.escape(name)}="
        r"(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan)",
        block,
    )
    if not match or match.group(1).lower() == "nan":
        return None
    return float(match.group(1))


metric_keys = [
    "success_once", "return", "returns_mean", "num_trajectories",
    "actor/approx_kl", "actor/clip_fraction", "actor/clipped_ratio",
    "actor/ratio", "actor/ratio_abs", "actor/policy_loss",
    "actor/policy_loss_abs", "actor/grad_norm", "actor/total_loss",
    "critic/value_loss", "critic/value_clip_ratio",
    "critic/explained_variance", "step", "generate_rollouts",
    "actor/run_training", "sync_weights", "env/env/bootstrap_step",
    "env/env_interact_step", "rollout/predict",
]

text = (ROOT / "metrics.log").read_text(encoding="utf-8", errors="replace")
rows = []
for block in re.split(r"(?=Global Step:)", text):
    match = re.search(r"Global Step:\s*(\d+)/(\d+)", block)
    if not match:
        continue
    row = {"global_step": int(match.group(1)), "max_steps": int(match.group(2))}
    for key in metric_keys:
        row[key] = parse_number(block, key)
    rows.append(row)

metrics = pd.DataFrame(rows).sort_values("global_step")
if metrics.empty:
    raise RuntimeError("No completed metric tables found")

resources = pd.read_csv(ROOT / "resources.csv")
resources["timestamp"] = pd.to_datetime(resources["timestamp"])
resources["elapsed_hours"] = (
    resources["timestamp"] - resources["timestamp"].iloc[0]
).dt.total_seconds() / 3600.0
for col in ["gpu0_memory_mb", "gpu1_memory_mb", "env_rss_mb", "actor_rss_mb", "rollout_rss_mb"]:
    resources[col.replace("_mb", "_gib")] = resources[col] / 1024.0

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 180,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.22,
})

steps = metrics["global_step"].to_numpy()
xmax = max(1, int(steps.max()))
xticks = np.arange(0, xmax + 1, max(1, int(np.ceil(xmax / 14))))


def format_step_axis(ax):
    ax.set_xlim(0, xmax)
    ax.set_xticks(xticks)
    ax.set_xlabel("Global step (starts at 0; no synthetic step-0 value)")


# 1. Success: one large plot for phone viewing.
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(steps, metrics["success_once"] * 100, "o-", lw=2.6, label="Success once")
ax.plot(steps, metrics["returns_mean"] * 100, "s--", lw=1.8, label="Mean GAE return x100")
ax.set_ylim(0, 100)
ax.set_ylabel("Percent")
format_step_axis(ax)
ax.set_title("Fast-WAM PPO - move_stapler_pad - completed history")
ax.legend()
fig.tight_layout()
fig.savefig(ROOT / "success_full.png", bbox_inches="tight")
plt.close(fig)


# 2. PPO actor and critic.
fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
axes[0, 0].plot(steps, metrics["actor/policy_loss"], "o-", label="Policy loss")
axes[0, 0].plot(steps, metrics["actor/total_loss"], "s--", label="Total loss")
axes[0, 0].axhline(0, color="#64748b", lw=1)
axes[0, 0].set_title("Actor losses")
axes[0, 0].legend()

axes[0, 1].plot(steps, metrics["critic/value_loss"], "o-", label="Value loss")
axes[0, 1].plot(steps, metrics["critic/value_clip_ratio"], "s--", label="Value clip ratio")
axes[0, 1].set_title("Critic metrics")
axes[0, 1].legend()

axes[1, 0].plot(steps, metrics["actor/approx_kl"], "o-", label="Approx KL")
axes[1, 0].plot(steps, metrics["actor/clip_fraction"], "s-", label="Clip fraction")
axes[1, 0].axhline(0, color="#64748b", lw=1)
axes[1, 0].set_title("PPO trust-region signals")
axes[1, 0].legend()

axes[1, 1].plot(steps, metrics["actor/ratio"], "o-", label="Ratio")
ax2 = axes[1, 1].twinx()
ax2.plot(steps, metrics["actor/grad_norm"], "s-", color="#9333ea", label="Grad norm")
axes[1, 1].axhline(1, color="#64748b", lw=1)
axes[1, 1].set_title("Ratio and pre-clip gradient norm")
lines = axes[1, 1].get_lines() + ax2.get_lines()
axes[1, 1].legend(lines, [line.get_label() for line in lines])
ax2.set_ylabel("Gradient norm")

for ax in axes.flat:
    format_step_axis(ax)
fig.suptitle("Fast-WAM PPO actor and critic - completed history", y=1.01, fontsize=16)
fig.tight_layout()
fig.savefig(ROOT / "ppo_actor_critic_full.png", bbox_inches="tight")
plt.close(fig)


# 3. Time, separating sequential totals from overlapping nested timers.
top = pd.DataFrame({
    "Rollout": metrics["generate_rollouts"],
    "Actor train": metrics["actor/run_training"],
    "Weight sync": metrics["sync_weights"],
}) / 60.0
top["Other"] = (metrics["step"] / 60.0 - top.sum(axis=1)).clip(lower=0)

fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
bottom = np.zeros(len(metrics))
for label, color in zip(top.columns, ["#2563eb", "#dc2626", "#16a34a", "#94a3b8"]):
    values = top[label].to_numpy()
    axes[0].bar(steps, values, bottom=bottom, label=label, color=color)
    bottom += values
axes[0].plot(steps, metrics["step"] / 60.0, "ko-", lw=1.4, label="Step wall time")
axes[0].set_ylabel("Minutes")
axes[0].set_title("Top-level sequential time decomposition")
axes[0].legend(ncol=3)

axes[1].plot(steps, metrics["env/env/bootstrap_step"] / 60.0, "o-", label="RoboTwin bootstrap/reset")
axes[1].plot(steps, metrics["env/env_interact_step"] / 60.0, "s-", label="RoboTwin action execution")
axes[1].plot(steps, metrics["rollout/predict"] / 60.0, "^-", label="Fast-WAM predict")
axes[1].plot(steps, metrics["actor/run_training"] / 60.0, "d-", label="Actor/critic train")
axes[1].set_ylabel("Minutes")
axes[1].set_title("Nested phase timers (overlap; do not sum)")
axes[1].legend(ncol=2)
for ax in axes:
    format_step_axis(ax)
fig.suptitle("Fast-WAM PPO time - completed history", y=1.01, fontsize=16)
fig.tight_layout()
fig.savefig(ROOT / "time_full.png", bbox_inches="tight")
plt.close(fig)


# 4. Full-resolution resources for phone viewing.
hours = resources["elapsed_hours"]
fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)
axes[0].plot(hours, resources["cgroup_ram_pct"], lw=1.3)
axes[0].axhline(90, color="#dc2626", ls="--", lw=1.1, label="90%")
axes[0].set_ylim(0, 102)
axes[0].set_ylabel("RAM (%)")
axes[0].set_title("Cgroup host RAM")
axes[0].legend()

axes[1].plot(hours, resources["gpu0_memory_gib"], label="GPU 0", lw=1.2)
axes[1].plot(hours, resources["gpu1_memory_gib"], label="GPU 1", lw=1.2)
axes[1].set_ylim(0, 80)
axes[1].set_ylabel("GiB")
axes[1].set_title("A800 memory")
axes[1].legend()

axes[2].plot(hours, resources["gpu0_util_pct"], label="GPU 0", lw=.8)
axes[2].plot(hours, resources["gpu1_util_pct"], label="GPU 1", lw=.8)
axes[2].set_ylim(0, 102)
axes[2].set_ylabel("Utilization (%)")
axes[2].set_title("Instantaneous GPU utilization")
axes[2].legend()

axes[3].plot(hours, resources["env_rss_gib"], label="Environment", lw=1.1)
axes[3].plot(hours, resources["actor_rss_gib"], label="Actor", lw=1.1)
axes[3].plot(hours, resources["rollout_rss_gib"], label="Rollout", lw=1.1)
axes[3].set_ylabel("RSS (GiB)")
axes[3].set_xlabel("Elapsed wall time (hours from monitor start)")
axes[3].set_title("Worker process RSS")
axes[3].legend(ncol=3)
for ax in axes:
    ax.set_xlim(0, hours.iloc[-1])
fig.suptitle("Fast-WAM PPO resources - full run history", y=1.0, fontsize=16)
fig.tight_layout()
fig.savefig(ROOT / "resources_full.png", bbox_inches="tight")
plt.close(fig)


def records(frame: pd.DataFrame):
    return json.loads(frame.to_json(orient="records"))


target = 360
index = np.linspace(0, len(resources) - 1, min(target, len(resources)), dtype=int)
resource_view = resources.iloc[index]
summary = {
    "metrics": records(metrics),
    "resources": records(resource_view[[
        "elapsed_hours", "cgroup_ram_pct", "gpu0_memory_gib", "gpu1_memory_gib",
        "gpu0_util_pct", "gpu1_util_pct", "env_rss_gib", "actor_rss_gib", "rollout_rss_gib",
    ]]),
    "stats": {
        "completed_steps": int(metrics["global_step"].max()),
        "latest_success": float(metrics["success_once"].iloc[-1]),
        "best_success": float(metrics["success_once"].max()),
        "mean_success": float(metrics["success_once"].mean()),
        "mean_step_minutes": float(metrics["step"].mean() / 60.0),
        "mean_rollout_minutes": float(metrics["generate_rollouts"].mean() / 60.0),
        "mean_actor_minutes": float(metrics["actor/run_training"].mean() / 60.0),
        "ram_current_pct": float(resources["cgroup_ram_pct"].iloc[-1]),
        "ram_peak_pct": float(resources["cgroup_ram_pct"].max()),
        "gpu0_current_gib": float(resources["gpu0_memory_gib"].iloc[-1]),
        "gpu1_current_gib": float(resources["gpu1_memory_gib"].iloc[-1]),
        "gpu0_peak_gib": float(resources["gpu0_memory_gib"].max()),
        "gpu1_peak_gib": float(resources["gpu1_memory_gib"].max()),
        "cgroup_oom": int(resources["cgroup_oom"].max()),
        "cgroup_oom_kill": int(resources["cgroup_oom_kill"].max()),
        "resource_hours": float(hours.iloc[-1]),
    },
}
(ROOT / "analysis_current.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary["stats"], indent=2))
