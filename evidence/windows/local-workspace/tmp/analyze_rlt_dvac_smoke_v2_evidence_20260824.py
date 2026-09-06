from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


root = Path(
    "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/"
    "rlt_dvac_real_smoke_8env1c_20260824_v2"
)


def q(values: np.ndarray, p: float) -> float:
    return float(np.quantile(values, p))


weights: list[np.ndarray] = []
z_values: list[np.ndarray] = []
npz_shapes: dict[str, list[int]] = {}
for path in sorted(root.glob("actor_rank*_update_*.npz")):
    with np.load(path, allow_pickle=False) as data:
        w = np.asarray(data["weights"], dtype=np.float64)
        z = np.asarray(data["z_scores"], dtype=np.float64)
        weights.append(w.reshape(-1))
        z_values.append(z.reshape(-1))
        npz_shapes[path.name] = list(w.shape)
w_matrix = np.concatenate([value.reshape(-1, 10) for value in weights], axis=0)
w_all = w_matrix.reshape(-1)
z_all = np.concatenate(z_values)
weight_sum = float(w_all.sum())
global_weight_ess = weight_sum**2 / (w_all.size * float(np.square(w_all).sum()))
top_n = max(1, int(round(0.2 * w_all.size)))
top20_mass = float(np.sort(w_all)[-top_n:].sum() / weight_sum)
row_sums = w_matrix.sum(axis=1)
per_query_ess = np.square(row_sums) / (
    w_matrix.shape[1] * np.square(w_matrix).sum(axis=1)
)
per_query_top20 = np.sort(w_matrix, axis=1)[:, -2:].sum(axis=1) / row_sums

with (root / "resources.csv").open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))


def max_float(name: str) -> float:
    return max(float(row[name]) for row in rows)


summary = {
    "trace_file_count": len(weights),
    "weight_count": int(w_all.size),
    "weight_shape_by_file": npz_shapes,
    "weight": {
        "min": float(w_all.min()),
        "p05": q(w_all, 0.05),
        "median": q(w_all, 0.5),
        "mean": float(w_all.mean()),
        "p95": q(w_all, 0.95),
        "max": float(w_all.max()),
        "global_ess_ratio": global_weight_ess,
        "global_top20_mass": top20_mass,
        "per_query_ess_mean": float(per_query_ess.mean()),
        "per_query_top20_mass_mean": float(per_query_top20.mean()),
        "mean_by_h": [float(value) for value in w_matrix.mean(axis=0)],
        "at_min_fraction": float(np.mean(w_all <= 1e-7)),
        "at_max_fraction": float(np.mean(w_all >= 2.0 - 1e-7)),
        "downweighted_fraction": float(np.mean(w_all < 1.0)),
        "upweighted_fraction": float(np.mean(w_all > 1.0)),
    },
    "z": {
        "mean": float(z_all.mean()),
        "std": float(z_all.std()),
        "min": float(z_all.min()),
        "max": float(z_all.max()),
    },
    "resource_samples": len(rows),
    "resource": {
        "gpu0_peak_mib": max_float("gpu0_used_mib"),
        "gpu1_peak_mib": max_float("gpu1_used_mib"),
        "cgroup_peak_gib": max_float("cgroup_current_bytes") / 2**30,
        "host_available_min_gib": min(float(row["host_available_bytes"]) for row in rows) / 2**30,
        "shm_peak_gib": max_float("shm_used_bytes") / 2**30,
        "oom_event_max": max_float("cgroup_oom_events"),
        "oom_kill_event_max": max_float("cgroup_oom_kill_events"),
    },
}
(root / "analysis_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2, sort_keys=True))
