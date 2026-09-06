from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path("tmp/rlt_dvac_amplitude_audit_20260827")
trace_root = ROOT / "all_traces"
FILES = sorted(glob.glob(str(trace_root / "update_*.npz")))
FILES = [path for path in FILES if Path(path).stat().st_size > 0]


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = left.reshape(-1).astype(np.float64)
    right = right.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denom) if denom > 0 else float("nan")


def angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    value = np.clip(cosine(left, right), -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


weights = []
z_scores = []
success = []
student = []
reference = []
executed = []
actor_switch = []
steps = []
for path in FILES:
    data = np.load(path)
    count = data["weights"].shape[0]
    weights.append(data["weights"].astype(np.float64))
    z_scores.append(data["z_scores"].astype(np.float64))
    success.append(data["episode_success"].reshape(count).astype(bool))
    student.append(data["student_actions"].astype(np.float64))
    reference.append(data["ref_chunk"].astype(np.float64))
    executed.append(data["executed_actions"].astype(np.float64))
    actor_switch.append(data["actor_switch"].reshape(count).astype(bool))
    steps.extend([int(data["update_step"][0])] * count)

w = np.concatenate(weights, axis=0)
z = np.concatenate(z_scores, axis=0)
succ = np.concatenate(success, axis=0)
pi = np.concatenate(student, axis=0)
ref = np.concatenate(reference, axis=0)
act = np.concatenate(executed, axis=0)
switched = np.concatenate(actor_switch, axis=0)
steps = np.asarray(steps)
active = np.any(np.abs(w - 1.0) > 1e-6, axis=-1)

# The method branch switches target and applies DVAC only after the frozen
# baseline is available. Uniform-w traces therefore belong to pre-apply warmup.
w = w[active]
z = z[active]
succ = succ[active]
pi = pi[active]
ref = ref[active]
act = act[active]
switched = switched[active]
steps = steps[active]

success_w = w[succ]
sum_w = success_w.sum(axis=-1)
sum_w2 = np.square(success_w).sum(axis=-1)
ess = np.square(sum_w) / (success_w.shape[-1] * sum_w2)
cv = success_w.std(axis=-1) / success_w.mean(axis=-1)
coefficient_angles = np.degrees(np.arccos(np.clip(np.sqrt(ess), 0.0, 1.0)))
sorted_w = np.sort(success_w, axis=-1)[:, ::-1]
top20_mass = sorted_w[:, :2].sum(axis=-1) / sorted_w.sum(axis=-1)
top10_mass = sorted_w[:, :1].sum(axis=-1) / sorted_w.sum(axis=-1)

# Per-coordinate BC-gradient proxies, omitting common positive constants.
g_ref = pi - ref
target_success = np.where(succ[:, None, None], act, ref)
g_success = pi - target_success
g_weighted = np.where(succ[:, None, None], w[:, :, None] * (pi - act), pi - ref)

succ_ref_err = np.square(pi[succ] - ref[succ]).mean(axis=(1, 2))
succ_exec_err = np.square(pi[succ] - act[succ]).mean(axis=(1, 2))
succ_weighted_err = (
    w[succ] * np.square(pi[succ] - act[succ]).mean(axis=-1)
).mean(axis=-1)
succ_exec_ref = np.square(act[succ] - ref[succ]).mean(axis=(1, 2))

result = {
    "source_files": len(FILES),
    "update_step_min": int(steps.min()),
    "update_step_max": int(steps.max()),
    "queries": int(len(succ)),
    "success_queries": int(succ.sum()),
    "success_ratio": float(succ.mean()),
    "success_weight_distribution": {
        "min": float(success_w.min()),
        "p05": float(np.quantile(success_w, 0.05)),
        "median": float(np.median(success_w)),
        "mean": float(success_w.mean()),
        "p95": float(np.quantile(success_w, 0.95)),
        "max": float(success_w.max()),
        "std": float(success_w.std()),
        "rms_deviation_from_one": float(np.sqrt(np.square(success_w - 1.0).mean())),
        "z_at_minus_2_ratio": float(np.isclose(z[succ], -2.0, atol=1e-6).mean()),
        "z_at_plus_2_ratio": float(np.isclose(z[succ], 2.0, atol=1e-6).mean()),
    },
    "redistribution": {
        "ess_mean": float(ess.mean()),
        "effective_h_of_10": float(10.0 * ess.mean()),
        "coefficient_angle_deg_mean": float(coefficient_angles.mean()),
        "cv_mean": float(cv.mean()),
        "top10_weight_mass_mean": float(top10_mass.mean()),
        "top20_weight_mass_mean": float(top20_mass.mean()),
    },
    "success_bc_losses": {
        "student_to_reference_mse_mean": float(succ_ref_err.mean()),
        "student_to_executed_mse_mean": float(succ_exec_err.mean()),
        "weighted_student_to_executed_mse_mean": float(succ_weighted_err.mean()),
        "executed_to_reference_mse_mean": float(succ_exec_ref.mean()),
        "weighting_loss_ratio_vs_unweighted_executed": float(
            succ_weighted_err.mean() / succ_exec_err.mean()
        ),
        "target_replacement_loss_ratio_executed_vs_reference": float(
            succ_exec_err.mean() / succ_ref_err.mean()
        ),
    },
    "bc_gradient_proxies_all_queries": {
        "target_replacement_angle_deg_ref_to_success": angle_deg(g_ref, g_success),
        "target_replacement_norm_ratio_success_over_ref": float(
            np.linalg.norm(g_success) / np.linalg.norm(g_ref)
        ),
        "dvac_weight_angle_deg_success_to_weighted": angle_deg(g_success, g_weighted),
        "dvac_weight_norm_ratio_weighted_over_success": float(
            np.linalg.norm(g_weighted) / np.linalg.norm(g_success)
        ),
        "combined_angle_deg_ref_to_weighted": angle_deg(g_ref, g_weighted),
        "combined_norm_ratio_weighted_over_ref": float(
            np.linalg.norm(g_weighted) / np.linalg.norm(g_ref)
        ),
    },
    "bc_gradient_proxies_success_only": {
        "target_replacement_angle_deg_ref_to_executed": angle_deg(
            g_ref[succ], g_success[succ]
        ),
        "target_replacement_norm_ratio_executed_over_ref": float(
            np.linalg.norm(g_success[succ]) / np.linalg.norm(g_ref[succ])
        ),
        "dvac_weight_angle_deg_unweighted_to_weighted": angle_deg(
            g_success[succ], g_weighted[succ]
        ),
        "dvac_weight_norm_ratio_weighted_over_unweighted": float(
            np.linalg.norm(g_weighted[succ]) / np.linalg.norm(g_success[succ])
        ),
    },
}

for label, mask in {
    "success_reference_collection": succ & ~switched,
    "success_student_collection": succ & switched,
}.items():
    if not mask.any():
        continue
    local_ref = pi[mask] - ref[mask]
    local_exec = pi[mask] - act[mask]
    local_weighted = w[mask, :, None] * local_exec
    result[label] = {
        "queries": int(mask.sum()),
        "student_to_reference_mse_mean": float(np.square(local_ref).mean()),
        "student_to_executed_mse_mean": float(np.square(local_exec).mean()),
        "executed_to_reference_mse_mean": float(
            np.square(act[mask] - ref[mask]).mean()
        ),
        "target_replacement_angle_deg": angle_deg(local_ref, local_exec),
        "target_replacement_norm_ratio": float(
            np.linalg.norm(local_exec) / np.linalg.norm(local_ref)
        ),
        "dvac_weight_angle_deg": angle_deg(local_exec, local_weighted),
        "dvac_weight_norm_ratio": float(
            np.linalg.norm(local_weighted) / np.linalg.norm(local_exec)
        ),
    }

(ROOT / "amplitude_summary.json").write_text(
    json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
)
print(json.dumps(result, indent=2, sort_keys=True))
