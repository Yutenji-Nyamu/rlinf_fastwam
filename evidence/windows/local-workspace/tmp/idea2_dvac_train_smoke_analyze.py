from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


run_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
output_dir.mkdir(parents=True, exist_ok=False)


def load_step(step: int) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = {}
    for rank_dir in sorted((run_dir / "dvac_train").glob("actor_rank*")):
        with np.load(rank_dir / f"rollout_step{step:04d}.npz", allow_pickle=False) as data:
            for key in data.files:
                chunks.setdefault(key, []).append(data[key])
    return {key: np.concatenate(values, axis=1) for key, values in chunks.items()}


step0 = load_step(0)
step1 = load_step(1)
weights = step1["weights"].reshape(-1, 50).astype(np.float64)
v1 = step1["v_l3"].reshape(-1, 50).astype(np.float64)

per_h_rows = []
for h in range(50):
    column = weights[:, h]
    log_v = np.log10(v1[:, h] + 1e-12)
    per_h_rows.append(
        {
            "h": h,
            "weight_p10": float(np.quantile(column, 0.10)),
            "weight_p25": float(np.quantile(column, 0.25)),
            "weight_median": float(np.median(column)),
            "weight_mean": float(column.mean()),
            "weight_p75": float(np.quantile(column, 0.75)),
            "weight_p90": float(np.quantile(column, 0.90)),
            "log10_v_l3_median": float(np.median(log_v)),
        }
    )

with (output_dir / "WEIGHT_BY_H.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(per_h_rows[0]))
    writer.writeheader()
    writer.writerows(per_h_rows)

match = (
    (step0["dvac_meta_source_env_rank"] == 0)
    & (step0["dvac_meta_local_env_slot"] == 0)
    & (step0["dvac_meta_rollout_epoch"] == 0)
    & (step0["dvac_meta_reset_id"] == 57)
    & (step0["dvac_meta_query_idx"] == 2)
)
locations = np.argwhere(match)
assert locations.shape == (1, 2), locations
time_index, batch_index = locations[0]
control_curve = step0["v_l3"][time_index, batch_index].astype(np.float64)

h = np.arange(50)
fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)

ax = axes[0]
ax.fill_between(
    h,
    [row["weight_p10"] for row in per_h_rows],
    [row["weight_p90"] for row in per_h_rows],
    color="#4c78a8",
    alpha=0.18,
    label="query p10–p90",
)
ax.plot(h, [row["weight_mean"] for row in per_h_rows], color="#4c78a8", label="mean")
ax.plot(
    h,
    [row["weight_median"] for row in per_h_rows],
    color="#f58518",
    label="median",
)
ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
ax.set(title="Runner step 2: DVAC gradient weights by future action", xlabel="future action h", ylabel="weight")
ax.set_ylim(0.79, 1.21)
ax.grid(alpha=0.25)
ax.legend()

ax = axes[1]
ax.plot(h, np.log10(control_curve + 1e-12), marker="o", markersize=3, color="#54a24b")
ax.axvline(10, color="#e45756", linestyle="--", label="first success: approximate h≈10")
ax.set(
    title="Recorded reset 57, q2: pre-execution DVAC curve",
    xlabel="future action h",
    ylabel=r"$\log_{10}(V_{L3}(h)+10^{-12})$",
)
ax.grid(alpha=0.25)
ax.legend()

figure_path = output_dir / "SMOKE_DVAC_WEIGHT_AND_CONTROL_ALIGNMENT.png"
fig.savefig(figure_path, dpi=180)
plt.close(fig)

summary = {
    "step2_queries": int(weights.shape[0]),
    "weight_min": float(weights.min()),
    "weight_mean": float(weights.mean()),
    "weight_median": float(np.median(weights)),
    "weight_max": float(weights.max()),
    "fraction_below_one": float((weights < 1.0).mean()),
    "fraction_above_one": float((weights > 1.0).mean()),
    "mean_weight_h_0_24": float(weights[:, :25].mean()),
    "mean_weight_h_25_49": float(weights[:, 25:].mean()),
    "control_query": {
        "reset_id": 57,
        "query_idx": 2,
        "action_slot_start": 100,
        "success_before": int(step0["dvac_meta_success_before"][time_index, batch_index]),
        "approx_first_success_h": 10,
        "v_l3_at_h10": float(control_curve[10]),
        "v_l3_max_h": int(np.argmax(control_curve)),
        "v_l3_min_h": int(np.argmin(control_curve)),
    },
    "figure": str(figure_path),
}
(output_dir / "SMOKE_ANALYSIS_SUMMARY.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, sort_keys=True))
print("IDEA2_TRAIN_SMOKE_ANALYSIS_PASS=1")
