"""Read-only action diagnostics for the released absolute-14D DW05 runtime.

Use the policy's *parsed* norm_stats, not an independently guessed JSON schema.
OpenDW e33befa dw05_policy.py:67-104 resolves global_mean/std to mean/std;
229-232 computes (x - mean) / (std + 1e-8), then clips to [-5, 5].
No input is modified. Command differences are not physical velocities.
"""

import numpy as np


JOINT_DIMS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)


def _range(values):
    return {"min": float(values.min()), "max": float(values.max()),
            "min_by_dim": values.min(axis=0).tolist(),
            "max_by_dim": values.max(axis=0).tolist()}


def _zscore_diagnostics(values, stats):
    # These are exactly the keys and float32 arithmetic used by the policy.
    if not isinstance(stats, dict) or not {"mean", "std"}.issubset(stats):
        return {"available": False, "reason": "parsed mean/std unavailable"}
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    if (mean.shape != (14,) or std.shape != (14,) or
            not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 0)):
        return {"available": False, "reason": "parsed mean/std are not finite 14D statistics"}
    denominator = std + 1.0e-8
    z = (values - mean) / denominator
    clipped_z = np.clip(z, -5.0, 5.0)
    clipped = np.abs(z) > 5.0
    # Physical-unit difference introduced by only the WM's z clipping.
    raw_clip_difference = np.abs(z - clipped_z) * denominator
    return {"available": True, "formula": "(raw-mean)/(std+1e-8)",
            "wm_clip_bounds": [-5.0, 5.0], "abs_z_gt5_count": int(clipped.sum()),
            "abs_z_gt5_count_by_dim": clipped.sum(axis=0).tolist(),
            "abs_z_gt5_fraction": float(clipped.mean()),
            "abs_z_max_by_dim": np.abs(z).max(axis=0).tolist(),
            "raw_clip_difference_max_by_dim": raw_clip_difference.max(axis=0).tolist()}


def action_telemetry(actions, state, *, norm_stats=None,
                     normalization_mode=None, action_condition_mode=None):
    """Summarize one raw C32 sample without changing policy/WM inputs."""
    actions = np.asarray(actions, dtype=np.float32)
    state = np.asarray(state, dtype=np.float32).reshape(1, 14)
    if actions.shape != (32, 14):
        raise ValueError("Telemetry expects the service's raw [32,14] actions")
    delta = np.abs(np.diff(actions, axis=0))
    result = {
        "raw_actions": _range(actions), "raw_state": _range(state),
        "first_command_abs_delta_from_state": np.abs(actions[0] - state[0]).tolist(),
        "adjacent_command_abs_delta": {
            "unit": "raw command units per annotation step; not per second",
            "p50_by_dim": np.quantile(delta, 0.50, axis=0).tolist(),
            "p95_by_dim": np.quantile(delta, 0.95, axis=0).tolist(),
            "p99_by_dim": np.quantile(delta, 0.99, axis=0).tolist(),
            "max_by_dim": delta.max(axis=0).tolist(),
            "joint_dims": list(JOINT_DIMS),
            "joint_p95": float(np.quantile(delta[:, JOINT_DIMS], 0.95)),
        },
        "normalization_mode": normalization_mode,
        "action_condition_mode": action_condition_mode,
    }
    if normalization_mode != "zscore_14d" or action_condition_mode != "absolute":
        result["wm_normalization"] = {"available": False, "reason": "diagnostics require zscore_14d absolute runtime"}
    else:
        parsed = norm_stats if isinstance(norm_stats, dict) else {}
        result["wm_normalization"] = {
            "actions": _zscore_diagnostics(actions, parsed.get("action")),
            "state": _zscore_diagnostics(state, parsed.get("state")),
        }
    return result
