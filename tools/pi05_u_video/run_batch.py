"""Native DV50 B16/H50/M10 collection, U10/5 and complete head video."""
import json
import os
import random
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/pi05_signals"))
from signal_math import noise_disagreement, ugrow_10_vs_5
from recording import Recorder
from noise_samples import collect_noise_samples


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))
    tmp.replace(path)


def array(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        return (value.float() if value.dtype == torch.bfloat16 else value).numpy()
    return np.asarray(value)


class Endpoints:
    def step(self, *_):
        return nullcontext()


def infer(model, obs, signals):
    with torch.inference_mode():
        return model.predict_action_batch(obs, mode="eval", compute_values=False,
                                          num_steps=10, return_dvac_telemetry=True,
                                          record_signals=signals, consistency_steps=5, noise=None)


def run(path):
    from omegaconf import OmegaConf
    from rlinf.models.embodiment.openpi import get_model
    from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv

    c = read(path)
    noise4 = c.get("noise_samples", 0) == 4
    assert c.get("noise_samples", 0) in (0, 4)
    assert c["num_envs"] == 16 and c["model"]["num_action_chunks"] == 50
    assert os.environ["CUDA_VISIBLE_DEVICES"] == os.environ["DOJO_GPU_SCOPE"] == str(c["gpu"])
    out = Path(c["output"])
    out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    save(out / "config.json", c)
    save(out / "started.json", {"time": start, "pid": os.getpid(), "gpu": c["gpu"]})
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.backends.cuda.enable_cudnn_sdp(False)
    model = get_model(OmegaConf.create(c["model"])).cuda().eval()
    model._signal_observer = Endpoints()
    dummy = {"main_images": torch.zeros((16, 224, 224, 3), dtype=torch.uint8),
             "wrist_images": torch.zeros((16, 2, 224, 224, 3), dtype=torch.uint8),
             "extra_view_images": None, "states": torch.zeros((16, 14)),
             "task_descriptions": [c["task"]] * 16}
    # Accepted model-before-SAPIEN initialization. Verify U once on each lane.
    cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
    a, result = infer(model, dummy, False)
    del result
    if c["acceptance"]:
        cpu_after, cuda_after = torch.get_rng_state(), torch.cuda.get_rng_state()
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state(cuda)
        b, result = infer(model, dummy, True)
        assert torch.equal(a, b)
        assert torch.equal(cpu_after, torch.get_rng_state()) and torch.equal(cuda_after, torch.cuda.get_rng_state())
        assert float(result["dvac_telemetry"]["side_final_time"]) == 0
        save(out / "u-parity.json", {"action_exact": True, "rng_exact": True, "side_final_time": 0})
        if noise4:
            observed = collect_noise_samples(model, dummy, result["dvac_telemetry"], b,
                                             c["noise_seed"] + 440000003)
            assert np.array_equal(observed["noise4_env_actions"][:, 0], array(b))
            assert torch.equal(cpu_after, torch.get_rng_state()) and torch.equal(cuda_after, torch.cuda.get_rng_state())
            assert observed["noise4_model_actions"].shape == (16, 4, 50, 14)
            assert noise_disagreement(observed["noise4_model_actions"]).valid.all()
            save(out / "noise4-parity.json", {"main_action_exact": True, "rng_exact": True,
                 "samples": 4, "extra_batch": 48, "num_steps": 10,
                 "cuda_peak_bytes": torch.cuda.max_memory_allocated()})
            del observed
        del b, result
    del dummy, a
    env, recorders = None, []
    success, counts = np.zeros(16, bool), np.zeros(16, int)
    queries, timings = [], []
    try:
        save(out / "progress.json", {"time": time.time(), "phase": "environment_init"})
        ec = OmegaConf.create(c["env"])
        assert not ec.auto_reset and ec.ignore_terminations and not ec.video_cfg.save_video
        env = RoboTwinEnv(ec, 16, 0, 1, None)
        obs, _ = env.reset()
        requested = env.reset_state_ids.cpu().tolist()
        actual = [int(e.task.ep_num) for e in env.venv.envs]
        seeds = [dict(slot=i, requested=int(requested[i]), actual=actual[i],
                      native_retry=int(requested[i]) != actual[i]) for i in range(16)]
        save(out / "seeds.json", seeds)
        for i, sub in enumerate(env.venv.envs):
            recorders.append(Recorder(sub.task, out, i, actual[i], fps=25))
        random.seed(c["noise_seed"])
        np.random.seed(c["noise_seed"])
        torch.manual_seed(c["noise_seed"])
        torch.cuda.manual_seed_all(c["noise_seed"])
        for q in range((c["step_limit"] + 49) // 50):
            active = ~success & (counts < c["step_limit"])
            if not active.any():
                break
            before = counts.copy()
            obs.setdefault("extra_view_images", None)
            input_obs = {k: array(obs[k]) for k in ("main_images", "wrist_images", "states")}
            descriptions = list(obs["task_descriptions"])
            save(out / "progress.json", {"time": time.time(), "phase": "inference", "query": q})
            tic = time.perf_counter()
            actions, result = infer(model, obs, True)
            trace = result["dvac_telemetry"]
            data = {k: array(trace[k]) for k in ("initial_noise_full", "final_model_action",
                    "side_action_model", "side_timesteps", "side_final_time", "x_chain")}
            score = ugrow_10_vs_5(data["final_model_action"], data["side_action_model"])
            assert score.valid.all()
            action = array(actions)
            assert action.shape == (16, 50, 14) and np.isfinite(action).all()
            data.update(env_action=action, u=score.score, u_valid=score.valid, active=active, before=before)
            if noise4:
                data.update(collect_noise_samples(model, obs, trace, actions,
                            c["noise_seed"] + 440000003 + 1009 * q))
                dispersion = noise_disagreement(data["noise4_model_actions"])
                assert dispersion.valid.all()
                data.update(noise4_u=dispersion.score, noise4_valid=dispersion.valid)
            inference_seconds = time.perf_counter() - tic
            np.savez_compressed(out / f"query_{q:03d}_obs.npz", **input_obs)
            # Signals survive a task error; completeness still requires done+validation.
            np.savez_compressed(out / f"query_{q:03d}.npz", **data)
            for recorder in recorders:
                recorder.begin(q)
            save(out / "progress.json", {"time": time.time(), "phase": "environment", "query": q})
            tic = time.perf_counter()
            obss, _, _, _, _ = env.chunk_step(action)
            obs = obss[-1]
            counts = np.array([int(e.task.take_action_cnt) for e in env.venv.envs])
            after = np.array([bool(e.task.eval_success) for e in env.venv.envs])
            for i, recorder in enumerate(recorders):
                recorder.end(after[i])
            data.update(after=counts, success_after=after)
            np.savez_compressed(out / f"query_{q:03d}.npz", **data)
            queries.append(dict(query=q, descriptions=descriptions, active=active.tolist(),
                                submitted_before=before.tolist(), submitted_after=counts.tolist(),
                                success_after=after.tolist()))
            timings.append(dict(query=q, inference_seconds=inference_seconds,
                                environment_and_video_seconds=time.perf_counter() - tic,
                                physics_steps=[r.ticks - r.query_start for r in recorders],
                                frames=[len(r.frames) for r in recorders]))
            success |= after
            save(out / "queries.json", queries)
            save(out / "timing.json", timings)
            save(out / "progress.json", {"time": time.time(), "phase": "saved", "query": q + 1})
            del actions, result, trace, data, input_obs
        assert not np.any(~success & (counts < c["step_limit"]))
        episodes = [dict(**seeds[i], task=c["task"], batch=c["batch"], success=bool(success[i]),
                         submitted=int(counts[i]), control_steps=recorders[i].ticks,
                         frames=len(recorders[i].frames), video=f"episode_{i:02d}.mp4",
                         frame_map=f"episode_{i:02d}_frames.json") for i in range(16)]
        save(out / "episodes.json", episodes)
    finally:
        if env is not None:
            env.venv.env_thread_pool.shutdown(wait=True)
        for recorder in recorders:
            recorder.close()
        if env is not None:
            env.offload(clear_cache=True)
    save(out / "done.json", dict(time=time.time(), elapsed_seconds=time.time() - start,
                                  episodes=16, successes=int(success.sum()), queries=len(queries),
                                  video="25Hz simulated timeline plus query endpoints; every control step mapped",
                                  u="same observation/full noise M10 vs M5, H50 D14",
                                  noise_samples=4 if noise4 else 0,
                                  noise_signal="population std / RMS across four N10 samples; mean D14" if noise4 else None))


if __name__ == "__main__":
    try:
        run(sys.argv[1])
    except BaseException:
        out = Path(read(sys.argv[1])["output"])
        if out.exists():
            save(out / "error.json", {"time": time.time(), "error": traceback.format_exc()})
        raise
