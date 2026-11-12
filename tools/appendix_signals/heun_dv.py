"""Paired Euler/Heun diagnostic on fixed recorded pi0.5 inputs (no training).

Heun follows the predictor/corrector pattern in NVlabs/edm/generate.py.
Unlike EDM's sigma division, this flow velocity is defined at t=0: all N
steps receive a correction, so N=10 costs exactly 20 velocity evaluations.
"""
import argparse
from contextlib import contextmanager
import gc
import json
import sys
from pathlib import Path
from types import MethodType

import numpy as np

from .common import Timer, cuda_guard, write_json, write_npz


@contextmanager
def heun_steps(model):
    """Only change this model instance's deterministic integration update."""
    original = model.sample_mean_var_val
    def corrected(self, x, idx, state, masks, cache, method, steps, compute_values=True):
        if method != "flow_ode" or compute_values:
            raise ValueError("Heun diagnostic is deterministic, inference-only")
        predictor, std, value, v1 = original(
            x, idx, state, masks, cache, method, steps, compute_values
        )
        grid = self._get_timesteps(steps, x.device)
        tnext = grid[idx + 1].expand(state.shape[0])
        v2, _ = self.get_velocity(state, predictor, tnext, masks, cache)
        dt = grid[idx + 1] - grid[idx]
        result = x + dt * (v1 + v2) * 0.5
        # Native telemetry receives v1 at the step start: one estimate per step.
        return result, std, value, v1
    model.sample_mean_var_val = MethodType(corrected, model)
    try:
        yield
    finally:
        model.sample_mean_var_val = original


def main(args):
    torch = cuda_guard()
    sys.path.insert(0, args.model_source)
    from omegaconf import OmegaConf
    from rlinf.models.embodiment.openpi import get_model
    from rlinf.algorithms.dvac_train_weighting import compute_endpoint_variance
    from tools.pi05_u_video.noise_samples import numpy_exact
    from tools.pi05_policy_eval.entropy_analysis import executed_count, reached_positions
    root, out = Path(args.data), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((root / "rt-pool/config.json").read_text())
    queries = json.loads((root / "rt-pool/queries.json").read_text())
    selected = [q for q in queries if q["task"] == args.task]
    assert len(selected) == 32
    with np.load(root / "rt-pool/observations.npz", allow_pickle=False) as z:
        obs = {k: z[k].copy() for k in z.files}
    with np.load(root / "rt-k8/bank_0.npz", allow_pickle=False) as z:
        initial = z["initial_noise"][[q["index"] for q in selected]].copy()
        seeds = z["seeds"][[q["index"] for q in selected]].copy()
    ids = np.repeat([q["index"] for q in selected], 8)
    masks, cache_reached = [], {}
    for q in selected:
        batch = Path(q["batch"])
        if str(batch) not in cache_reached:
            cache_reached[str(batch)] = reached_positions(batch)[1]
        with np.load(batch / f"query_{q['query']:03d}.npz", allow_pickle=False) as z:
            s = q["slot"]
            count = executed_count(cache_reached[str(batch)], s, q["query"],
                                   z["before"][s], z["after"][s], z["active"][s])
        masks.append(np.arange(50) < count)
    model = get_model(OmegaConf.create(cfg["model"])).cuda().eval()
    model.requires_grad_(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    def pack(indices, source=obs):
        result = {k: torch.from_numpy(v[indices].copy()) for k, v in source.items()
                  if k != "task_descriptions"}
        result["task_descriptions"] = source["task_descriptions"][indices].tolist()
        result["extra_view_images"] = None
        return result
    def predict(ob, noise):
        with torch.inference_mode():
            actions, result = model.predict_action_batch(
                ob, noise=noise.cuda().to(model.action_in_proj.weight.dtype),
                mode="eval", compute_values=False, num_steps=10,
                return_dvac_telemetry=True, record_signals=False)
        trace = result["dvac_telemetry"]
        answer = {k: numpy_exact(trace[k]) for k in
                  ["x_chain", "z_endpoint", "final_model_action", "timesteps"]}
        answer["env_action"] = numpy_exact(actions)
        answer["dv"] = numpy_exact(compute_endpoint_variance(trace["z_endpoint"], 5))
        return answer
    # Exact replay of an existing B16 native Euler sample before changing solver.
    batch = Path(selected[0]["batch"])
    with np.load(batch / "query_000_obs.npz", allow_pickle=False) as z:
        old = {k: z[k].copy() for k in z.files}
    qm = json.loads((batch / "queries.json").read_text())
    if isinstance(qm, dict): qm = qm["queries"]
    old["task_descriptions"] = np.asarray(qm[0]["descriptions"])
    with np.load(batch / "query_000.npz", allow_pickle=False) as z:
        old_noise = z["noise4_initial_noise"][:, 0].copy()
        target = z["noise4_model_actions"][:, 0].copy()
        target_env = z["noise4_env_actions"][:, 0].copy()
    replay = predict(pack(np.arange(len(old_noise)), old), torch.from_numpy(old_noise))
    gate = dict(model_max_abs=float(np.max(np.abs(replay["final_model_action"]-target))),
                env_max_abs=float(np.max(np.abs(replay["env_action"]-target_env))))
    write_json(out / "replay-gate.json", gate)
    np.testing.assert_allclose(replay["final_model_action"], target, atol=1e-5, rtol=0)
    np.testing.assert_allclose(replay["env_action"], target_env, atol=1e-5, rtol=0)
    del replay
    gc.collect(); torch.cuda.empty_cache()
    noise = torch.from_numpy(initial.reshape(-1, 50, 32))
    write_npz(out / "inputs.npz", initial_noise=initial, seeds=seeds,
              executed_mask=np.asarray(masks),
              **{k: v[[q["index"] for q in selected]] for k,v in obs.items()})
    write_json(out / "queries.json", [{k:v for k,v in q.items() if k != "batch"} for q in selected])
    timings = {}
    # Previously measured B256 peak <62 GiB reserved. Same shapes for both solvers;
    # Heun doubles sequential calls, not resident model copies or batch size.
    for solver in ["euler", "heun"]:
        calls = [0]
        original_velocity = model.get_velocity
        def counted(*a, **kw):
            calls[0] += 1
            return original_velocity(*a, **kw)
        model.get_velocity = counted
        try:
            with Timer() as timer:
                if solver == "heun":
                    with heun_steps(model): result = predict(pack(ids), noise)
                else:
                    result = predict(pack(ids), noise)
        finally:
            model.get_velocity = original_velocity
        assert calls[0] == (10 if solver == "euler" else 20), calls
        assert all(np.isfinite(v).all() for v in result.values())
        if solver == "euler":
            first = result["z_endpoint"][:, 0].copy()
        else:
            np.testing.assert_array_equal(result["z_endpoint"][:, 0], first)
        write_npz(out / f"{solver}.npz", **result)
        timings[solver] = dict(**timer.result, velocity_calls=calls[0], batch=len(ids))
        write_json(out / "timings.json", timings)
        write_json(out / "progress.json", dict(solver=solver, state="complete"))
    write_json(out / "complete.json", dict(
        state="complete", task=args.task, observations=32, episodes=8,
        noise_samples_per_observation=8, paired_chunks=256, N=10, L=5, H=50, D=14,
        dv="FP32 population variance of last 5 clean-action estimates, sum over D14",
        clean_action="step-start x-t*v1; one estimate per full step for both solvers",
        inference="frozen SFT pi0.5; deterministic ODE; same observations/noise/grid/dtype",
        scope="offline paired signal comparison; no new rollout success-rate estimate",
        checkpoint=Path(cfg["model"]["model_path"]).name, timings=timings,
        source_commit=cfg["source_commit"],
        references=["https://github.com/NVlabs/edm/blob/main/generate.py",
                    "https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model-source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--task", default="turn_switch")
    main(p.parse_args())
