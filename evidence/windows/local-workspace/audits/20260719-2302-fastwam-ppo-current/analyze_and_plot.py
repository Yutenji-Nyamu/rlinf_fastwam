from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
METRICS = ROOT / "metrics.log"
RESOURCES = ROOT / "resources.csv"


def metric(block: str, name: str):
    matches = re.findall(
        rf"(?<![\w/]){re.escape(name)}=([^\s│]+)", block
    )
    if not matches:
        return None
    raw = matches[-1].rstrip("s")
    if raw.lower() == "nan":
        return None
    try:
        return float(raw)
    except ValueError:
        return raw


text = METRICS.read_text(encoding="utf-8", errors="replace")
rows = []
for block in re.split(r"(?=Global Step:)", text):
    match = re.search(r"Global Step:\s*(\d+)/(\d+)", block)
    if not match:
        continue
    row = {
        "global_step": int(match.group(1)),
        "max_steps": int(match.group(2)),
    }
    for key in [
        "success_once",
        "return",
        "returns_mean",
        "num_trajectories",
        "actor/approx_kl",
        "actor/clip_fraction",
        "actor/ratio",
        "actor/ratio_abs",
        "actor/policy_loss",
        "actor/policy_loss_abs",
        "actor/grad_norm",
        "actor/total_loss",
        "critic/value_loss",
        "critic/value_clip_ratio",
        "critic/explained_variance",
        "step",
    ]:
        row[key] = metric(block, key)
    rows.append(row)

if not rows:
    raise RuntimeError("No completed metric tables found")

metrics = pd.DataFrame(rows).sort_values("global_step")
resources = pd.read_csv(RESOURCES)
resources["timestamp"] = pd.to_datetime(resources["timestamp"])
resources["elapsed_hours"] = (
    resources["timestamp"] - resources["timestamp"].iloc[0]
).dt.total_seconds() / 3600.0
resources["gpu0_gib"] = resources["gpu0_memory_mb"] / 1024.0
resources["gpu1_gib"] = resources["gpu1_memory_mb"] / 1024.0

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 180,
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.22,
    }
)

steps = metrics["global_step"].to_numpy()
xmax = max(int(steps.max()), 1)

# Mobile-friendly success chart.
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(
    steps,
    metrics["success_once"] * 100,
    marker="o",
    linewidth=2.6,
    color="#0f766e",
    label="Success once",
)
ax.plot(
    steps,
    metrics["returns_mean"] * 100,
    marker="s",
    linewidth=2.0,
    color="#7c3aed",
    label="Mean PPO return ×100",
)
ax.set_xlim(0, xmax)
ax.set_ylim(0, 100)
ax.set_xticks(np.arange(0, xmax + 1, 1))
ax.set_xlabel("Global step (axis begins at 0; no synthetic step-0 point)")
ax.set_ylabel("Percent")
ax.set_title("Fast-WAM PPO · move_stapler_pad · full completed history")
ax.legend(loc="best")
fig.tight_layout()
fig.savefig(ROOT / "success_full.png", bbox_inches="tight")
plt.close(fig)

# Mobile-friendly optimization chart.
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
ax = axes[0, 0]
ax.plot(steps, metrics["actor/policy_loss"], marker="o", color="#2563eb")
ax.axhline(0, color="#64748b", linewidth=1)
ax.set_title("Actor policy loss")
ax.set_ylabel("Loss")

ax = axes[0, 1]
ax.plot(steps, metrics["critic/value_loss"], marker="o", color="#dc2626")
ax.set_title("Critic value loss")
ax.set_ylabel("Loss")

ax = axes[1, 0]
ax.plot(steps, metrics["actor/approx_kl"], marker="o", label="Approx KL", color="#ea580c")
ax.plot(steps, metrics["actor/clip_fraction"], marker="s", label="Clip fraction", color="#0891b2")
ax.axhline(0, color="#64748b", linewidth=1)
ax.set_title("PPO trust-region signals")
ax.set_ylabel("Value")
ax.legend(loc="best")

ax = axes[1, 1]
ax.plot(steps, metrics["actor/ratio"], marker="o", label="Ratio", color="#16a34a")
ax2 = ax.twinx()
ax2.plot(steps, metrics["actor/grad_norm"], marker="s", label="Grad norm (pre-clip)", color="#9333ea")
ax.axhline(1, color="#64748b", linewidth=1)
ax.set_title("Ratio and gradient norm")
ax.set_ylabel("Ratio")
ax2.set_ylabel("Grad norm")
lines = ax.get_lines() + ax2.get_lines()
ax.legend(lines, [line.get_label() for line in lines], loc="best")

for ax in axes.flat:
    ax.set_xlim(0, xmax)
    ax.set_xticks(np.arange(0, xmax + 1, 1))
for ax in axes[1, :]:
    ax.set_xlabel("Global step (no synthetic step-0 point)")
fig.suptitle("Fast-WAM PPO optimization · full completed history", y=1.01, fontsize=17)
fig.tight_layout()
fig.savefig(ROOT / "optimization_full.png", bbox_inches="tight")
plt.close(fig)

# Full resource timeline from the first monitor sample.
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
axes[0].plot(resources["elapsed_hours"], resources["cgroup_ram_pct"], color="#b45309", linewidth=1.5)
axes[0].axhline(90, color="#dc2626", linestyle="--", linewidth=1.2, label="90%")
axes[0].set_ylim(0, 102)
axes[0].set_ylabel("Cgroup RAM (%)")
axes[0].set_title("Host memory")
axes[0].legend(loc="best")

axes[1].plot(resources["elapsed_hours"], resources["gpu0_gib"], label="GPU 0", color="#2563eb", linewidth=1.4)
axes[1].plot(resources["elapsed_hours"], resources["gpu1_gib"], label="GPU 1", color="#0f766e", linewidth=1.4)
axes[1].set_ylim(0, 80)
axes[1].set_ylabel("GPU memory (GiB)")
axes[1].set_xlabel("Elapsed wall time (hours, from monitor start)")
axes[1].set_title("A800 memory")
axes[1].legend(loc="best")
for ax in axes:
    ax.set_xlim(0, resources["elapsed_hours"].iloc[-1])
fig.suptitle("Fast-WAM PPO resources · full run history", y=1.01, fontsize=17)
fig.tight_layout()
fig.savefig(ROOT / "resources_full.png", bbox_inches="tight")
plt.close(fig)

# Downsample only for the interactive rendering; extrema are preserved separately.
target = 240
if len(resources) > target:
    idx = np.linspace(0, len(resources) - 1, target, dtype=int)
    resource_view = resources.iloc[idx]
else:
    resource_view = resources

def clean_records(frame: pd.DataFrame):
    return json.loads(frame.to_json(orient="records"))

summary = {
    "metrics": clean_records(metrics),
    "resources": clean_records(
        resource_view[
            [
                "elapsed_hours",
                "cgroup_ram_pct",
                "gpu0_gib",
                "gpu1_gib",
                "gpu0_util_pct",
                "gpu1_util_pct",
            ]
        ]
    ),
    "snapshot": {
        "completed_steps": int(metrics["global_step"].max()),
        "latest_success": float(metrics["success_once"].iloc[-1]),
        "best_success": float(metrics["success_once"].max()),
        "mean_success": float(metrics["success_once"].mean()),
        "latest_value_loss": float(metrics["critic/value_loss"].iloc[-1]),
        "first_value_loss": float(metrics["critic/value_loss"].iloc[0]),
        "latest_approx_kl": float(metrics["actor/approx_kl"].iloc[-1]),
        "latest_clip_fraction": float(metrics["actor/clip_fraction"].iloc[-1]),
        "mean_step_minutes": float(metrics["step"].mean() / 60.0),
        "resource_hours": float(resources["elapsed_hours"].iloc[-1]),
        "ram_current_pct": float(resources["cgroup_ram_pct"].iloc[-1]),
        "ram_peak_pct": float(resources["cgroup_ram_pct"].max()),
        "ram_p95_pct": float(resources["cgroup_ram_pct"].quantile(0.95)),
        "ram_over_90_fraction": float((resources["cgroup_ram_pct"] >= 90).mean()),
        "gpu0_current_gib": float(resources["gpu0_gib"].iloc[-1]),
        "gpu1_current_gib": float(resources["gpu1_gib"].iloc[-1]),
        "gpu0_peak_gib": float(resources["gpu0_gib"].max()),
        "gpu1_peak_gib": float(resources["gpu1_gib"].max()),
        "cgroup_oom": int(resources["cgroup_oom"].max()),
        "cgroup_oom_kill": int(resources["cgroup_oom_kill"].max()),
    },
}
(ROOT / "analysis.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps(summary["snapshot"], indent=2))
