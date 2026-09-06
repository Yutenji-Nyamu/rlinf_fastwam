from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np


files = [
    path
    for path in sorted(
        glob.glob(
            "tmp/rlt_dvac_amplitude_audit_20260827/all_traces/update_*.npz"
        )
    )
    if Path(path).stat().st_size > 0
][-20:]


def angle(left: np.ndarray, right: np.ndarray) -> float:
    left = left.reshape(-1).astype(np.float64)
    right = right.reshape(-1).astype(np.float64)
    cosine = np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def cat(key: str) -> np.ndarray:
    return np.concatenate([np.load(path)[key] for path in files], axis=0)


w = cat("weights").astype(np.float64)
succ = cat("episode_success").reshape(-1).astype(bool)
switch = cat("actor_switch").reshape(-1).astype(bool)
pi = cat("student_actions").astype(np.float64)
ref = cat("ref_chunk").astype(np.float64)
act = cat("executed_actions").astype(np.float64)

g_ref = pi - ref
target = np.where(succ[:, None, None], act, ref)
g_success = pi - target
g_weighted = np.where(succ[:, None, None], w[:, :, None] * (pi - act), pi - ref)
sw = w[succ]
ess = sw.sum(-1) ** 2 / (sw.shape[1] * np.square(sw).sum(-1))

result = {
    "trace_files": len(files),
    "first_update": int(np.load(files[0])["update_step"][0]),
    "last_update": int(np.load(files[-1])["update_step"][0]),
    "queries": int(len(succ)),
    "success_queries": int(succ.sum()),
    "success_ratio": float(succ.mean()),
    "success_student_queries": int((succ & switch).sum()),
    "success_weight_p05_mean_p95": [
        float(np.quantile(sw, 0.05)),
        float(sw.mean()),
        float(np.quantile(sw, 0.95)),
    ],
    "success_weight_ess": float(ess.mean()),
    "success_top20_mass": float(
        np.mean(np.sort(sw, axis=-1)[:, -2:].sum(-1) / sw.sum(-1))
    ),
    "success_weight_coefficient_angle_deg": float(
        np.mean(np.degrees(np.arccos(np.clip(np.sqrt(ess), 0.0, 1.0))))
    ),
    "all_query_target_replacement_angle_deg": angle(g_ref, g_success),
    "all_query_dvac_weight_angle_deg": angle(g_success, g_weighted),
    "success_only_target_replacement_angle_deg": angle(
        g_ref[succ], g_success[succ]
    ),
    "success_only_dvac_weight_angle_deg": angle(
        g_success[succ], g_weighted[succ]
    ),
    "success_only_target_replacement_norm_ratio": float(
        np.linalg.norm(g_success[succ]) / np.linalg.norm(g_ref[succ])
    ),
    "success_only_dvac_weight_norm_ratio": float(
        np.linalg.norm(g_weighted[succ]) / np.linalg.norm(g_success[succ])
    ),
}
print(json.dumps(result, indent=2, sort_keys=True))
