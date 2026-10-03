"""Inference-only N16/H50/M10 signal batch, retaining native DV50 env behavior."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np

import signal_math as sm


SCHEMA = "pi05-action-signals-v1"
SIGNALS = ("dv", "fresco", "geo", "geoaac", "shift", "norm", "sr", "ugrow")
MAIN_STEPS, SIDE_STEPS, HORIZON, BATCH, ACTION_DIM = 10, 5, 50, 16, 14


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)


def numpy_array(value, *, floating=False):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if floating or str(value.dtype) == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
    array = np.asarray(value)
    return np.array(array, dtype=np.float32 if floating else None, copy=True)


def compute_scores(data):
    """Shared CPU scorer; source arrays are also the persisted replay inputs."""
    return {
        "dv": sm.dv_score(data["z_endpoint"], tail_n=5),
        "fresco": sm.fresco_score(data["x_chain"][:, :MAIN_STEPS], tail_n=5),
        "geo": sm.geo_full(data["velocity"]),
        "geoaac": sm.geoaac_growth(data["velocity"], data["timesteps"]),
        "shift": sm.shift_score(data["shift_first_mean"], data["shift_last_mean"]),
        "norm": sm.norm_score(data["layer_norms"], tail_n=5, deepest_layers=3),
        "sr": sm.sr_window5(data["sr_last_hidden"]),
        "ugrow": sm.ugrow_10_vs_5(data["final_model_action"], data["side_action_model"]),
    }


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rng_snapshot(torch):
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (state[0], state[1].copy(), state[2], state[3], state[4]),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()],
    }


def _rng_restore(torch, state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def _rng_matches(torch, left, right):
    a, b = left["numpy"], right["numpy"]
    return {
        "python": left["python"] == right["python"],
        "numpy": a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:],
        "torch_cpu": bool(torch.equal(left["torch_cpu"], right["torch_cpu"])),
        "torch_cuda": len(left["torch_cuda"]) == len(right["torch_cuda"]) and all(
            torch.equal(x, y) for x, y in zip(left["torch_cuda"], right["torch_cuda"])),
    }


def _exact(torch, left, right):
    if torch.is_tensor(left) and torch.is_tensor(right):
        return left.dtype == right.dtype and left.shape == right.shape and bool(torch.equal(left, right))
    return np.array_equal(numpy_array(left), numpy_array(right))


def _observer_arrays(observed):
    hidden = observed["last_layer_hidden"]
    with np.errstate(over="ignore", invalid="ignore"):
        compressed_hidden = hidden.astype(np.float16)
    if not np.isfinite(compressed_hidden).all():
        raise FloatingPointError("Float16 hidden storage would introduce nonfinite values")
    result = {key: np.array(value, copy=True) for key, value in observed.items()
              if key not in {"last_layer_hidden", "shift_score", "norm_score"}}
    result["last_layer_hidden"] = compressed_hidden
    # Keep exact scoring input; all-round FP16 storage alone cannot reproduce
    # the unrounded FP32 SR calculation exactly.
    result["sr_last_hidden"] = np.array(hidden[:, -1], dtype=np.float32, copy=True)
    return result


def _payload(trace, observed):
    required = ("x_chain", "z_endpoint", "velocity", "timesteps", "final_model_action",
                "initial_noise_full", "side_action_model", "side_num_steps",
                "side_timesteps", "side_final_time")
    missing = set(required) - set(trace)
    if missing:
        raise RuntimeError(f"Missing signal model telemetry: {sorted(missing)}")
    data = {name: numpy_array(trace[name], floating=name != "side_num_steps")
            for name in required}
    data.update(_observer_arrays(observed))
    expected = {
        "x_chain": (BATCH, MAIN_STEPS + 1, HORIZON, ACTION_DIM),
        "z_endpoint": (BATCH, MAIN_STEPS, HORIZON, ACTION_DIM),
        "velocity": (BATCH, MAIN_STEPS, HORIZON, ACTION_DIM),
        "timesteps": (MAIN_STEPS,),
        "final_model_action": (BATCH, HORIZON, ACTION_DIM),
        "side_action_model": (BATCH, HORIZON, ACTION_DIM),
        "side_timesteps": (SIDE_STEPS,),
    }
    for key, shape in expected.items():
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise RuntimeError(f"Invalid {key}: expected finite {shape}, got {data[key].shape}")
    if data["initial_noise_full"].shape[:2] != (BATCH, HORIZON) or data["initial_noise_full"].ndim != 3:
        raise RuntimeError("Expected full initial model noise [16,50,Dmodel]")
    if not np.isfinite(data["initial_noise_full"]).all() or int(data["side_num_steps"]) != SIDE_STEPS:
        raise RuntimeError("Invalid initial noise or side_num_steps")
    if float(data["side_final_time"]) != 0.0:
        raise RuntimeError("Side solver did not report its t=0 endpoint")
    if not np.array_equal(data["initial_noise_full"][..., :ACTION_DIM], data["x_chain"][:, 0]):
        raise RuntimeError("Saved full initial noise does not match initial main latent")
    if not np.array_equal(data["x_chain"][:, -1], data["final_model_action"]):
        raise RuntimeError("Main chain endpoint differs from final model action")
    if not np.all(data["hook_counts"] == 1) or data["hook_counts"].shape[0] != MAIN_STEPS:
        raise RuntimeError("Observer hooks are missing, duplicated, or include the side chain")
    scores = compute_scores(data)
    for name, result in scores.items():
        data[name] = result.score
        data[name + "_valid"] = result.valid
    return data


def model_smoke(model, observer, torch):
    """One B16 parity check per batch; natural noise sampling on both passes."""
    def dummy():
        return {"main_images": torch.zeros((BATCH, 224, 224, 3), dtype=torch.uint8),
                "wrist_images": torch.zeros((BATCH, 2, 224, 224, 3), dtype=torch.uint8),
                "extra_view_images": None, "states": torch.zeros((BATCH, ACTION_DIM)),
                "task_descriptions": ["adjust the bottle"] * BATCH}

    before = _rng_snapshot(torch)
    with torch.inference_mode():
        baseline_action, baseline = model.predict_action_batch(
            dummy(), mode="eval", compute_values=False, return_dvac_telemetry=True,
            record_signals=False, noise=None)
    torch.cuda.synchronize()
    baseline_after = _rng_snapshot(torch)
    _rng_restore(torch, before)
    with torch.inference_mode():
        enabled_action, enabled = model.predict_action_batch(
            dummy(), mode="eval", compute_values=False, return_dvac_telemetry=True,
            record_signals=True, consistency_steps=SIDE_STEPS, noise=None)
    torch.cuda.synchronize()
    enabled_after = _rng_snapshot(torch)
    observed = observer.flush()
    equality = {"env_action": _exact(torch, baseline_action, enabled_action)}
    for key in ("x_chain", "z_endpoint", "timesteps", "final_model_action"):
        equality[key] = _exact(torch, baseline["dvac_telemetry"][key], enabled["dvac_telemetry"][key])
    baseline_chains = baseline.get("forward_inputs", {}).get("chains")
    enabled_chains = enabled.get("forward_inputs", {}).get("chains")
    if baseline_chains is None or enabled_chains is None:
        raise RuntimeError("Parity smoke requires original result['forward_inputs']['chains']")
    equality["full_model_chains"] = _exact(torch, baseline_chains, enabled_chains)
    rng_equal = _rng_matches(torch, baseline_after, enabled_after)
    if not all(equality.values()) or not all(rng_equal.values()):
        raise RuntimeError(f"Signal recording changes main inference/RNG: {equality}, {rng_equal}")
    data = _payload(enabled["dvac_telemetry"], observed)
    schedule = numpy_array(model._get_timesteps(SIDE_STEPS, next(model.parameters()).device), floating=True)
    if schedule.shape != (SIDE_STEPS + 1,) or float(schedule[-1]) != 0.0:
        raise RuntimeError("Side solver schedule must take five transitions to t=0")
    if float(data["side_final_time"]) != 0.0:
        raise RuntimeError("Side chain did not report a t=0 endpoint")
    report = {"time": time.time(), "batch": BATCH, "main_steps": MAIN_STEPS,
              "side_steps": SIDE_STEPS, "action_horizon": HORIZON,
              "main_exact_parity": equality, "rng_after_equal": rng_equal,
              "hook_counts_shape": list(data["hook_counts"].shape),
              "hook_counts_all_one": bool(np.all(data["hook_counts"] == 1)),
              "side_schedule": schedule.tolist(), "side_schedule_final_time": float(schedule[-1]),
              "side_reported_final_time": float(data["side_final_time"]),
              "side_final_action_finite": bool(np.isfinite(data["side_action_model"]).all()),
              "input_noise_method": "natural sampling with full RNG reset, avoiding supplied-noise dtype cast",
              "cudnn_sdpa": False, "note": "Synthetic runtime check; excluded from episode data"}
    _rng_restore(torch, baseline_after)
    return report


def _observation_arrays(obs):
    arrays = {key: numpy_array(obs[key]) for key in ("main_images", "wrist_images", "states")}
    # Preserve native camera pixels; the model owns any input resizing.
    main, wrist = arrays["main_images"], arrays["wrist_images"]
    if (main.ndim != 4 or main.shape[0] != BATCH or main.shape[-1] != 3
            or min(main.shape[1:3]) <= 0 or main.dtype != np.uint8):
        raise RuntimeError("Expected native uint8 main images [16,H,W,3] with positive H/W")
    if (wrist.ndim != 5 or wrist.shape[:2] != (BATCH, 2) or wrist.shape[-1] != 3
            or min(wrist.shape[2:4]) <= 0 or wrist.dtype != np.uint8):
        raise RuntimeError("Expected native uint8 wrist images [16,2,Hw,Ww,3] with positive Hw/Ww")
    if arrays["states"].shape != (BATCH, ACTION_DIM) or not np.isfinite(arrays["states"]).all():
        raise RuntimeError("Expected finite native state [16,14]")
    if obs.get("extra_view_images") is not None:
        arrays["extra_view_images"] = numpy_array(obs["extra_view_images"])
    return arrays


def run(config_path):
    import cv2
    import torch
    from omegaconf import OmegaConf
    from rlinf.models.embodiment.openpi import get_model
    from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
    from signal_observer import SignalObserver

    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "done.json").exists() or (out / "started.json").exists():
        raise RuntimeError("Existing started/completed batch requires a fresh output directory")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(cfg["gpu"]):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must exactly match the assigned GPU")
    if cfg["num_envs"] != BATCH or cfg.get("selected_l", 5) != 5 or cfg["step_limit"] < 1:
        raise ValueError("This runner fixes N16 and tail N5 and requires a positive full episode limit")
    started = time.time()
    save(out / "started.json", {"time": started, "pid": os.getpid(), "gpu": cfg["gpu"], "schema": SCHEMA})
    save(out / "config.json", cfg)
    save(out / "schema.json", {"schema": SCHEMA, "signals": list(SIGNALS),
        "shape": {"B": BATCH, "M": MAIN_STEPS, "H": HORIZON, "D": ACTION_DIM},
        "tail_n": 5, "norm_deepest_layers": 3, "sr_window": 5, "ugrow_steps": [10, 5],
        "hidden_storage": "last_layer_hidden float16: storage rounding only; sr_last_hidden float32 preserves scoring input",
        "shift_norm_storage": "shift layer means and per-layer L2 norms are float32",
        "score_storage": "float64 scores; bool valid masks; NaN iff invalid",
        "fresco_input": "x_chain[:,:10]; excludes post-update final x_chain[:,10]",
        "geoaac_first_action": "NaN/False: no previous nonempty prefix",
        "signal_weights": "raw scores only; no reward/loss weighting and no environment intervention",
        "cleanup_order": "threadpool.shutdown(wait=True), then env.offload(clear_cache=True)",
        "source_hashes": {name: _sha(Path(__file__).with_name(name)) for name in
                          ("run_batch.py", "signal_math.py", "signal_observer.py", "validate_batch.py")}})
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.backends.cuda.enable_cudnn_sdp(False)
    model = get_model(OmegaConf.create(cfg["model"])).cuda().eval()
    if model.config.num_steps != MAIN_STEPS or model.config.action_horizon != HORIZON:
        raise RuntimeError("Model configuration must retain native M10/H50")
    observer = SignalObserver(model, main_steps=MAIN_STEPS)
    model._signal_observer = observer
    env = None
    videos = []
    frame_times = [[] for _ in range(BATCH)]
    success = np.zeros(BATCH, dtype=bool)
    counts = np.zeros(BATCH, dtype=int)
    query_rows = []
    queries_done = 0
    try:
        save(out / "model-smoke.json", model_smoke(model, observer, torch))
        env_cfg = OmegaConf.create(cfg["env"])
        if env_cfg.auto_reset or not env_cfg.ignore_terminations:
            raise RuntimeError("Native DV50 requires auto_reset=False and ignore_terminations=True")
        env = RoboTwinEnv(env_cfg, BATCH, 0, 1, None)
        obs, _ = env.reset()
        requested = env.reset_state_ids.cpu().tolist()
        actual = [int(sub.task.ep_num) for sub in env.venv.envs]
        seeds = [{"slot": i, "requested": int(requested[i]), "actual": actual[i],
                  "native_retry": int(requested[i]) != actual[i]} for i in range(BATCH)]
        save(out / "seeds.json", seeds)
        random.seed(cfg["noise_seed"])
        np.random.seed(cfg["noise_seed"])
        torch.manual_seed(cfg["noise_seed"])
        torch.cuda.manual_seed_all(cfg["noise_seed"])
        heads = numpy_array(obs["main_images"])
        height, width = heads.shape[1:3]
        for slot in range(BATCH):
            path = out / f"episode_{slot:02d}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open video writer {path}")
            videos.append(writer)

        def snapshot(images):
            for slot, image in enumerate(images):
                videos[slot].write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                frame_times[slot].append(time.time())

        snapshot(heads)
        for query in range((cfg["step_limit"] + HORIZON - 1) // HORIZON):
            active = ~success & (counts < cfg["step_limit"])
            if not active.any():
                break
            before = counts.copy()
            obs.setdefault("extra_view_images", None)
            obs_arrays = _observation_arrays(obs)
            descriptions = list(obs["task_descriptions"])
            if len(descriptions) != BATCH or not all(isinstance(item, str) for item in descriptions):
                raise RuntimeError("Expected 16 task descriptions")
            save(out / "progress.json", {"time": time.time(), "query": query, "phase": "model",
                "successes": int(success.sum()), "queries_max": (cfg["step_limit"] + 49) // 50})
            with torch.inference_mode():
                actions, result = model.predict_action_batch(
                    obs, mode="eval", compute_values=False, return_dvac_telemetry=True,
                    record_signals=True, consistency_steps=SIDE_STEPS, noise=None)
            trace = result["dvac_telemetry"]
            observed = observer.flush()
            data = _payload(trace, observed)
            action = numpy_array(actions)
            if action.shape != (BATCH, HORIZON, ACTION_DIM) or not np.isfinite(action).all():
                raise RuntimeError("Invalid environment action")
            # Preserve the native complete-chunk call, including inactive slots.
            # TOPP interpolation does not provide one-to-one physical action slots.
            obss, _, _, _, _ = env.chunk_step(action)
            obs = obss[-1]
            counts = np.array([int(sub.task.take_action_cnt) for sub in env.venv.envs])
            after_success = np.array([bool(sub.task.eval_success) for sub in env.venv.envs])
            submitted = active[:, None] & (np.arange(HORIZON)[None, :] < (counts - before)[:, None])
            exact = submitted & ~after_success[:, None]
            snapshot(numpy_array(obs["main_images"]))
            data.update(env_action=action, submitted_mask=submitted, executed_mask=exact,
                        active=active, success_after=after_success,
                        action_slot_start=before, action_slot_end=counts)
            stem = out / f"query_{query:03d}"
            save_npz(stem.with_suffix(".npz"), **data)
            save_npz(out / f"query_{query:03d}_obs.npz", **obs_arrays)
            invalid = {name: int((~data[name + "_valid"]).sum()) for name in SIGNALS}
            save(stem.with_suffix(".json"), {"schema": SCHEMA, "query": query, "time": time.time(),
                "task_descriptions": descriptions, "extra_view_images_none": "extra_view_images" not in obs_arrays,
                "input_observation": f"query_{query:03d}_obs.npz", "trace": f"query_{query:03d}.npz",
                "trace_source_dtypes": {key: str(value.dtype) for key, value in trace.items() if hasattr(value, "dtype")},
                "invalid_score_counts": invalid, "pre_frame": query, "post_frame": query + 1,
                "last_layer_hidden_storage_dtype": "float16", "sr_input_storage_dtype": "float32"})
            for slot in range(BATCH):
                if not active[slot]:
                    continue
                values = data["dv"][slot, submitted[slot]]
                query_rows.append({"slot": slot, "query": query,
                    "action_slot_start": int(before[slot]), "action_slot_end": int(counts[slot]),
                    "dv_mean": float(values.mean()) if values.size else None,
                    "submitted": int(submitted[slot].sum()), "executed_prefix_known": not bool(after_success[slot]),
                    "success_after": bool(after_success[slot]), "pre_frame": query, "post_frame": query + 1,
                    "wall_time": time.time()})
            success |= after_success
            queries_done += 1
            save(out / "queries.json", query_rows)
            save(out / "progress.json", {"time": time.time(), "query": queries_done, "phase": "saved",
                 "queries_max": (cfg["step_limit"] + 49) // 50, "successes": int(success.sum()),
                 "invalid_score_counts": invalid})
            del result, trace, actions, observed, data, obs_arrays
        if np.any(~success & (counts < cfg["step_limit"])):
            raise RuntimeError("Loop ended before all native episodes completed")
        episodes = [{"slot": slot, "task": cfg["task"], "batch": cfg["batch"], **seeds[slot],
            "seed_source": cfg["seed_source"], "noise_seed": cfg["noise_seed"],
            "success": bool(success[slot]), "action_slots_submitted": int(counts[slot]),
            "termination": "success" if success[slot] else "step_limit",
            "video": f"episode_{slot:02d}.mp4", "frames": len(frame_times[slot]),
            "video_sampling": "head camera before/after each policy query; 4 fps preview",
            "physical_action_prefix": "unknown for terminal success chunk (TOPP interpolation)"}
            for slot in range(BATCH)]
        save(out / "episodes.json", episodes)
        save(out / "frame_times.json", frame_times)
    finally:
        for writer in videos:
            writer.release()
        try:
            if env is not None:
                # The old runner did offload before shutdown. Native close()
                # closes SubEnvs synchronously; first joining workers avoids
                # freeing simulator objects while a worker can still use them.
                env.venv.env_thread_pool.shutdown(wait=True)
                env.offload(clear_cache=True)
        finally:
            observer.close()
            model._signal_observer = None
    done = {"time": time.time(), "elapsed_s": time.time() - started, "schema": SCHEMA,
            "episodes": BATCH, "successes": int(success.sum()), "queries": queries_done,
            "source_commit": cfg["source_commit"], "validation": "pending CPU validation"}
    save(out / "done.json", done)
    print("SIGNALS_BATCH_DONE " + json.dumps(done), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    args = parser.parse_args()
    try:
        run(args.config)
    except BaseException:
        try:
            cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
            out = Path(cfg["output"])
            out.mkdir(parents=True, exist_ok=True)
            # Never overwrite completed-batch evidence on an accidental rerun.
            if not (out / "done.json").exists():
                save(out / "error.json", {"time": time.time(), "error": traceback.format_exc()})
        finally:
            raise


if __name__ == "__main__":
    main()
