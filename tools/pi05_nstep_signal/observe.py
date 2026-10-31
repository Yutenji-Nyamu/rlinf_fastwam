"""Observe solver budgets on one unchanged N10 trajectory and four shared noises."""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/pi05_u_video"))
sys.path.insert(0, str(ROOT / "tools/pi05_signals"))
from noise_samples import collect_noise_samples, numpy_exact, repeated_observation
from signal_math import dv_score, noise_disagreement, ugrow_10_vs_5

BUDGETS = (1, 5, 10)


def derive(data):
    """Recomputable action-indexed scores; N1 DV is explicitly unavailable."""
    out = {}
    for n in BUDGETS:
        values = np.asarray(data[f"actions_model_n{n}"], dtype=np.float64)[..., :14]
        assert values.ndim == 4 and values.shape[1:] == (4, 50, 14)
        for count, sample in [(4, values), (3, values[:, 1:])]:
            score = noise_disagreement(sample)
            out[f"noise{count}_relative_n{n}"] = score.score
            out[f"noise{count}_absolute_n{n}"] = sample.std(axis=1).mean(axis=-1)
            out[f"noise{count}_valid_n{n}"] = score.valid
        if n == 1:
            shape = (values.shape[0], 50)
            out["dv_n1"] = np.full(shape, np.nan)
            out["dv_valid_n1"] = np.zeros(shape, dtype=bool)
        else:
            score = dv_score(data[f"z_endpoint_n{n}"], tail_n=5)
            out[f"dv_n{n}"] = score.score
            out[f"dv_valid_n{n}"] = score.valid
    for a, b in [(1, 10), (1, 5), (5, 10)]:
        aa = np.asarray(data[f"actions_model_n{a}"][:, 0, :, :14], dtype=np.float64)
        bb = np.asarray(data[f"actions_model_n{b}"][:, 0, :, :14], dtype=np.float64)
        score = ugrow_10_vs_5(aa, bb)  # Same two-endpoint statistic for each budget pair.
        out[f"u_{a}_{b}"] = score.score
        out[f"u_valid_{a}_{b}"] = score.valid
        out[f"rmse_{a}_{b}"] = np.sqrt(np.mean((aa - bb) ** 2, axis=-1))
    return out


def observe(model, obs, main_trace, main_actions, seed, max_extra_batch=None):
    """Reuse the executed pass; all additional inference restores CPU/CUDA RNG."""
    initial = main_trace["initial_noise_full"]
    batch, horizon, dim = initial.shape
    assert (horizon, dim) == (50, 32)
    size = max_extra_batch or 3 * batch
    assert size > 0
    devices = [initial.device.index] if initial.device.type == "cuda" else []
    data = collect_noise_samples(model, obs, main_trace, main_actions, seed, size)
    # Preserve the validated sample-major ordering and exactly the same noises.
    extras = torch.as_tensor(
        data["noise4_initial_noise"][:, 1:].transpose(1, 0, 2, 3).reshape(3 * batch, 50, 32),
        device=initial.device, dtype=initial.dtype,
    )
    repeated = repeated_observation(obs, 3)
    data.update(
        solver_budgets=np.asarray(BUDGETS, dtype=np.int64),
        executed_steps=np.asarray(10, dtype=np.int64),
        actions_model_n10=data["noise4_model_actions"],
        actions_env_n10=data["noise4_env_actions"],
    )

    def keep_trace(n, trace):
        assert torch.equal(trace["initial_noise_full"], initial)
        for key in ("x_chain", "z_endpoint", "timesteps"):
            data[f"{key}_n{n}"] = numpy_exact(trace[key])
        grid = model._get_timesteps(n, initial.device)
        assert float(grid[-1]) == 0.0
        data[f"final_time_n{n}"] = numpy_exact(grid[-1])

    keep_trace(10, main_trace)
    with torch.random.fork_rng(devices=devices), torch.inference_mode():
        for n in (1, 5):
            action, result = model.predict_action_batch(
                obs, mode="eval", compute_values=False, num_steps=n,
                return_dvac_telemetry=True, record_signals=False, noise=initial.clone(),
            )
            trace = result["dvac_telemetry"]
            keep_trace(n, trace)
            main_model = numpy_exact(trace["final_model_action"])
            main_env = numpy_exact(action)
            model_parts, env_parts = [], []
            del result, trace, action
            for start in range(0, 3 * batch, size):
                sliced = {k: None if v is None else v[start:start + size] for k, v in repeated.items()}
                action, result = model.predict_action_batch(
                    sliced, mode="eval", compute_values=False, num_steps=n,
                    return_dvac_telemetry=True, record_signals=False,
                    noise=extras[start:start + size],
                )
                model_parts.append(numpy_exact(result["dvac_telemetry"]["final_model_action"]))
                env_parts.append(numpy_exact(action))
                del action, result
            for kind, main, parts in [("model", main_model, model_parts), ("env", main_env, env_parts)]:
                extra = np.concatenate(parts).reshape(3, batch, 50, 14).transpose(1, 0, 2, 3)
                data[f"actions_{kind}_n{n}"] = np.concatenate([main[:, None], extra], axis=1)
    data.update(derive(data))
    # Legacy aliases keep existing curve/export tools usable for N10 vs N5.
    data.update(
        initial_noise_full=numpy_exact(initial),
        final_model_action=data["actions_model_n10"][:, 0],
        side_action_model=data["actions_model_n5"][:, 0],
        u=data["u_5_10"], u_valid=data["u_valid_5_10"],
        noise4_u=data["noise4_relative_n10"], noise4_valid=data["noise4_valid_n10"],
        dv=data["dv_n10"],
    )
    return data
