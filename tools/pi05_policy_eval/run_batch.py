# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Paired B32 native evaluation, reusing the accepted 20261003 signal path."""

import argparse
import gc
import json
import os
import random
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
from protocol import (
    choose,
    configure_recording_wait,
    execution_lengths,
    read,
    recording_mode,
    save,
    validate_config,
    variable_chunk_step,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/pi05_signals"))
from signal_math import ugrow_10_vs_5  # noqa: E402

sys.path.insert(0, str(ROOT / "tools/pi05_u_video"))
from recording import Recorder  # noqa: E402


class EndpointOnly:
    """The accepted sampler can collect endpoints without attention hooks."""

    def step(self, *_):
        return nullcontext()


def arr(x):
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def infer(model, obs, steps, signals=False, telemetry=True, fql_features=False):
    with torch.inference_mode():
        return model.predict_action_batch(
            obs,
            mode="eval",
            compute_values=False,
            num_steps=steps,
            return_dvac_telemetry=telemetry,
            record_signals=signals,
            consistency_steps=5,
            noise=None,
            return_fql_features=fql_features,
        )


def run(config, model_cache=None):
    from omegaconf import OmegaConf

    from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
    from rlinf.models.embodiment.openpi import get_model

    c = read(config)
    validate_config(c)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == str(c["gpu"])
    assert os.environ["DOJO_GPU_SCOPE"] == str(c["gpu"])
    out = Path(c["output"])
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    save(out / "config.json", c)
    save(out / "started.json", {"time": started, "pid": os.getpid(), "gpu": c["gpu"]})
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.backends.cuda.enable_cudnn_sdp(False)
    policy, batch = c["policy"], c["num_envs"]
    takeover = c.get("takeover_mode", "weights")
    mode = recording_mode(c)
    full_video = mode == "full"
    record_decisions = c.get("record_decisions", True)
    collect = c.get("collect_transitions", False)
    shadow = c.get("teacher_shadow", False)
    diagnose_q = bool(c.get("fql_q_path"))
    if shadow:
        assert policy == "weak" and full_video and c["num_steps"] == 10
        assert c["execute_chunk"] == 20 and c["weak_weights"] and c["strong_weights"]
        assert not collect and not diagnose_q and model_cache is None
        from teacher_shadow import observe as observe_teacher, save_observation
    q_evaluation = None
    policy_metadata = None
    writer = None
    if collect:
        if c.get("suite") == "pi05_rlinf_single_distill":
            assert c["execute_chunk"] == 50 and c["num_steps"] == 5
        else:
            assert c["execute_chunk"] == 10 and c["num_steps"] == 10
        assert policy == "weak" and not any(
            c.get(k) for k in ("distill_weights", "weak_weights", "strong_weights")
        )
        assert mode == "none" and not record_decisions
        assert not c.get("collect_weak_u", False)
    signals = policy in ("calibration", "high", "low", "random") or c.get(
        "collect_weak_u", False
    ) or shadow
    need_strong = (
        (policy in ("strong", "high", "low", "random") and takeover == "weights") or shadow
    )
    cache_key = json.dumps(
        [
            c["model"],
            c.get("distill_weights"),
            c.get("weak_weights"),
            batch,
            c["num_steps"],
        ],
        sort_keys=True,
    )
    reused = model_cache is not None and model_cache.get("key") == cache_key
    if model_cache is not None:
        assert policy == "weak" and not signals and not need_strong
    load_started = time.perf_counter()
    weak = (
        model_cache["model"]
        if reused
        else (
            None
            if policy == "strong" and not c.get("collect_weak_u", False)
            else get_model(OmegaConf.create(c["model"]))
        )
    )
    strong = None
    if weak is not None and not reused:
        if c.get("distill_weights"):
            from rlinf.algorithms.pi05_distillation.model import load_expert

            state = torch.load(
                c["distill_weights"], map_location="cpu", weights_only=True
            )
            load_expert(weak, state["expert"])
            policy_metadata = {k: state[k] for k in ("step", "base")}
            save(
                out / "distill-loaded.json",
                {"path": c["distill_weights"], "method": state["method"]},
            )
            del state
        if c.get("weak_weights"):
            state = torch.load(c["weak_weights"], map_location="cpu", weights_only=True)
            loaded = weak.load_state_dict(state, strict=True)
            del state
            save(
                out / "weak-loaded.json",
                {
                    "path": c["weak_weights"],
                    "strict": True,
                    "missing_keys": list(loaded.missing_keys),
                    "unexpected_keys": list(loaded.unexpected_keys),
                },
            )
        weak = weak.cuda().eval()
        weak._signal_observer = EndpointOnly()
    if reused:
        policy_metadata = model_cache.get("policy_metadata")
    if model_cache is not None:
        model_cache.update(key=cache_key, model=weak, policy_metadata=policy_metadata)
    if diagnose_q:
        from rlinf.algorithms.pi05_distillation.q_evaluation import QEvaluation

        assert policy == "weak" and not collect and policy_metadata is not None
        q_evaluation = QEvaluation(c["fql_q_path"], c, torch.device("cuda:0"), policy_metadata)
    load_seconds = time.perf_counter() - load_started
    if need_strong:
        # Same base model/wrappers/normalization; use the rollout worker's
        # strict full-state overlay, never checkpoint-directory auto loading.
        strong = get_model(OmegaConf.create(c["model"]))
        state = torch.load(c["strong_weights"], map_location="cpu", weights_only=True)
        loaded = strong.load_state_dict(state, strict=True)
        del state
        strong = strong.cuda().eval()
        save(
            out / "strong-loaded.json",
            {
                "path": c["strong_weights"],
                "strict": True,
                "missing_keys": list(loaded.missing_keys),
                "unexpected_keys": list(loaded.unexpected_keys),
            },
        )
    for model in (weak, strong):
        if model is not None:
            assert (
                model.config.action_horizon == 50 and model.config.action_env_dim == 14
            )
    env, videos, recorders = None, [], []
    success, counts = np.zeros(batch, bool), np.zeros(batch, int)
    decisions, costs, frames = [], [], [[] for _ in range(batch)]
    threshold = read(c["threshold_path"]) if policy in ("high", "low") else {}
    try:
        warm_started = time.perf_counter()
        # Accepted order: warm the model before creating any SAPIEN scenes.
        dummy = {
            "main_images": torch.zeros((batch, 224, 224, 3), dtype=torch.uint8),
            "wrist_images": torch.zeros((batch, 2, 224, 224, 3), dtype=torch.uint8),
            "extra_view_images": None,
            "states": torch.zeros((batch, 14)),
            "task_descriptions": [
                c.get("warmup_prompt", "move the pillbottle to the pad")
            ]
            * batch,
        }
        for model in () if reused else (weak, strong):
            if model is not None:
                action, result = infer(model, dummy, c["num_steps"])
                assert tuple(action.shape) == (batch, 50, 14)
                del action, result
        if policy == "calibration" or shadow:
            # Once per card calibration verifies this lightweight observer preserves
            # the native main action and RNG. No simulator action is submitted.
            cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
            a, _ = infer(weak, dummy, 10)
            cpu_after, cuda_after = torch.get_rng_state(), torch.cuda.get_rng_state()
            torch.set_rng_state(cpu)
            torch.cuda.set_rng_state(cuda)
            b, trace = infer(weak, dummy, 10, True)
            assert np.array_equal(arr(a), arr(b))
            assert torch.equal(cpu_after, torch.get_rng_state())
            assert torch.equal(cuda_after, torch.cuda.get_rng_state())
            assert float(trace["dvac_telemetry"]["side_final_time"]) == 0
            save(
                out / "u-parity.json",
                {"action_exact": True, "rng_exact": True, "side_final_time": 0},
            )
            if shadow:
                trials = []
                for extra_batch in c.get("benchmark_noise_batches", [c.get("noise_extra_batch") or 3*batch]):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    benchmark_start = time.perf_counter()
                    try:
                        data = observe_teacher(weak, strong, dummy, trace["dvac_telemetry"], b,
                                               c["noise_seed"] + 7000003, extra_batch)
                        torch.cuda.synchronize()
                        trials.append({"extra_batch": extra_batch,
                                       "seconds": time.perf_counter()-benchmark_start,
                                       "peak_gib": torch.cuda.max_memory_allocated()/2**30})
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        trials.append({"extra_batch": extra_batch, "oom": True})
                accepted = [t for t in trials if not t.get("oom") and t["peak_gib"] < 77]
                assert accepted, "No accepted extra-noise batch size"
                selected_trial = min(accepted, key=lambda t: t["seconds"])
                c["noise_extra_batch"] = selected_trial["extra_batch"]
                save(out / "noise-capacity.json", {"environment_batch": batch, "trials": trials,
                                                   "selected_extra_batch": c["noise_extra_batch"]})
                save(out / "config.json", c)
                assert np.array_equal(data["noise4_env_actions"][:, 0].astype(np.float32), arr(b))
                assert torch.equal(cpu_after, torch.get_rng_state())
                assert torch.equal(cuda_after, torch.cuda.get_rng_state())
                save(out / "shadow-parity.json", {
                    "main_action_exact": True, "rng_exact": True, "same_teacher_noise": True,
                    "batch": batch, "extra_batch": c.get("noise_extra_batch") or 3*batch,
                    "peak_gib": selected_trial["peak_gib"],
                })
                del data
            del a, b, trace
        if diagnose_q:
            cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
            a, _ = infer(weak, dummy, c["num_steps"], telemetry=False)
            cpu_after, cuda_after = torch.get_rng_state(), torch.cuda.get_rng_state()
            torch.set_rng_state(cpu); torch.cuda.set_rng_state(cuda)
            b, result = infer(weak, dummy, c["num_steps"], telemetry=False, fql_features=True)
            assert np.array_equal(arr(a), arr(b))
            values = q_evaluation.predict(result["fql_features"], result["model_actions"], np.zeros(batch, int), c["step_limit"])
            assert np.isfinite(values).all()
            assert torch.equal(cpu_after, torch.get_rng_state()) and torch.equal(cuda_after, torch.cuda.get_rng_state())
            save(out / "q-parity.json", {"action_exact": True, "rng_exact": True, "finite_q": True})
            del a, b, result, values
        del dummy
        warm_seconds = time.perf_counter() - warm_started
        ec = OmegaConf.create(c["env"])
        assert not ec.auto_reset and ec.ignore_terminations
        env_started = time.perf_counter()
        env = RoboTwinEnv(ec, batch, 0, 1, None)
        built_at = time.perf_counter()
        obs, _ = env.reset()
        initialized_at = time.perf_counter()
        if full_video or "environment_step_timeout_seconds" in c:
            configure_recording_wait(
                env, seconds=c.get("environment_step_timeout_seconds", 300)
            )
        requested = env.reset_state_ids.cpu().tolist()
        actual = [int(e.task.ep_num) for e in env.venv.envs]
        assert len(set(requested)) == batch
        seeds = [
            {
                "slot": i,
                "requested": int(requested[i]),
                "actual": actual[i],
                "native_retry": int(requested[i]) != actual[i],
            }
            for i in range(batch)
        ]
        save(out / "seeds.json", seeds)
        if collect:
            from rlinf.algorithms.pi05_distillation.transitions import TransitionWriter

            writer = TransitionWriter(out, c, seeds, obs)
        save(
            out / "initialization.json",
            {
                "model_reused": reused,
                "load_seconds": load_seconds,
                "warmup_seconds": warm_seconds,
                "environment_total_seconds": initialized_at - env_started,
                "build_seconds": built_at - env_started,
                "reset_seconds": initialized_at - built_at,
                "instructions": obs.get("task_descriptions"),
                "states": arr(obs["states"]).tolist(),
            },
        )
        random.seed(c["noise_seed"])
        np.random.seed(c["noise_seed"])
        heads = arr(obs["main_images"]).astype(np.uint8)
        height, width = heads.shape[1:3]
        for slot in range(batch):
            if mode == "none":
                continue
            if full_video:
                recorders.append(
                    Recorder(env.venv.envs[slot].task, out, slot, actual[slot], fps=25)
                )
                continue
            writer = cv2.VideoWriter(
                str(out / f"episode_{slot:02d}.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                4,
                (width, height),
            )
            assert writer.isOpened()
            videos.append(writer)

        def snapshot(images):
            if mode != "sparse":
                return
            for slot, image in enumerate(images):
                videos[slot].write(
                    cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
                )
                frames[slot].append(time.time())

        snapshot(heads)
        chunk = c["execute_chunk"]
        min_chunk = 20 if takeover == "c20" else chunk
        for query in range((c["step_limit"] + min_chunk - 1) // min_chunk):
            active = ~success & (counts < c["step_limit"])
            if not active.any():
                break
            before = counts.copy()
            obs.setdefault("extra_view_images", None)
            save(
                out / "progress.json",
                {"time": time.time(), "phase": "inference", "query": query},
            )
            # Same full batch/slot noise for each matching decision across
            # conditions. Different solver step counts cannot shift next RNG.
            noise_seed = c["noise_seed"] + 1009 * query
            torch.manual_seed(noise_seed)
            torch.cuda.manual_seed(noise_seed)
            torch.cuda.synchronize()
            tic = time.perf_counter()
            u = None
            signal_data = None
            if weak is not None:
                actions, result = infer(
                    weak, obs, c["num_steps"], signals, telemetry=not collect, fql_features=diagnose_q
                )
                if diagnose_q:
                    q_values = q_evaluation.predict(result["fql_features"], result["model_actions"], counts, c["step_limit"])
                if collect:
                    normalized_actions = arr(result["model_actions"])
                if signals:
                    trace = result["dvac_telemetry"]
                    score = ugrow_10_vs_5(
                        arr(trace["final_model_action"]),
                        arr(trace["side_action_model"]),
                    )
                    assert score.valid.all()
                    u = score.score.mean(axis=1)
                    save(out / f"u_{query:03d}.json", score.score.tolist())
                    if full_video:
                        signal_data = {
                            k: arr(trace[k])
                            for k in (
                                "initial_noise_full",
                                "final_model_action",
                                "side_action_model",
                            )
                        }
                        signal_data.update(u=score.score, active=active, before=before)
                    if shadow:
                        signal_data.update(observe_teacher(weak, strong, obs, trace, actions,
                                                           noise_seed + 7000003, c.get("noise_extra_batch")))
                        save_observation(out / f"observation_{query:03d}.npz", obs)
                    del trace
                del result
            selected = choose(
                policy,
                u,
                active,
                threshold,
                np.random.default_rng(c["gate_seed"] + query).random(batch),
                query=query,
                max_takeover_queries=c.get("max_takeover_queries"),
            )
            if strong is not None and selected.any():
                # Full fixed batch retains slot identity and same natural noise.
                # Isolate the strong pass from the weak policy RNG stream.
                with torch.random.fork_rng(devices=[0]):
                    torch.manual_seed(noise_seed)
                    torch.cuda.manual_seed(noise_seed)
                    strong_actions, result = infer(strong, obs, 10)
                actions = (
                    strong_actions
                    if weak is None
                    else torch.where(
                        torch.as_tensor(selected, device=actions.device)[:, None, None],
                        strong_actions,
                        actions,
                    )
                )
                del strong_actions, result
            action = arr(actions)
            assert action.shape == (batch, 50, 14) and np.isfinite(action).all()
            torch.cuda.synchronize()
            inference_seconds = time.perf_counter() - tic
            lengths = execution_lengths(
                selected, active, counts, chunk, c["step_limit"], takeover
            )
            if signal_data is not None:
                signal_data.update(
                    env_action=action, selected=selected, offered=lengths
                )
                np.savez_compressed(out / f"query_{query:03d}.npz", **signal_data)
            save(
                out / "progress.json",
                {"time": time.time(), "phase": "environment", "query": query},
            )
            tic = time.perf_counter()
            for recorder in recorders:
                recorder.begin(query)
            if collect:
                # Same whole C10 native call, capturing unmodified native returns.
                # Do not split into single steps or infer terminal states from
                # the wrapper, which intentionally ignores terminations in eval.
                length = min(chunk, c["step_limit"] - query * chunk)
                raw_obs, raw_reward, native_term, native_trunc, _ = env.venv.step(
                    action[:, :length]
                )
                obs = env._extract_obs_image(raw_obs)
            elif takeover == "c20":
                obs = variable_chunk_step(env, action, lengths)
            else:
                length = min(chunk, c["step_limit"] - query * chunk)
                obss, _, _, _, _ = env.chunk_step(action[:, :length])
                obs = obss[-1]
            counts = np.array([int(e.task.take_action_cnt) for e in env.venv.envs])
            after = np.array([bool(e.task.eval_success) for e in env.venv.envs])
            assert np.all(counts <= c["step_limit"])
            if diagnose_q:
                q_evaluation.record(q_values, actual, query, active, before, counts, success, after, c["step_limit"])
            if collect:
                writer.step(
                    obs,
                    action,
                    normalized_actions,
                    active,
                    before,
                    counts,
                    success,
                    after,
                    (raw_reward, native_term, native_trunc),
                    query,
                    noise_seed,
                )
            for slot, recorder in enumerate(recorders):
                recorder.end(after[slot])
            if signal_data is not None:
                signal_data.update(after=counts, success_after=after)
                np.savez_compressed(out / f"query_{query:03d}.npz", **signal_data)
            if mode == "sparse":
                snapshot(arr(obs["main_images"]))
            for slot in np.flatnonzero(active) if record_decisions else []:
                decisions.append(
                    {
                        "slot": int(slot),
                        "query": query,
                        "actual_seed": actual[slot],
                        "u": float(u[slot]) if u is not None else None,
                        "strong": bool(selected[slot]),
                        "takeover_mode": takeover,
                        "eligible": c.get("max_takeover_queries") is None
                        or query < c["max_takeover_queries"],
                        "start": int(before[slot]),
                        "end": int(counts[slot]),
                        "offered_actions": int(lengths[slot]),
                        "success_after": bool(after[slot]),
                        "noise_seed": noise_seed,
                    }
                )
            success |= after
            costs.append(
                {
                    "query": query,
                    "inference_seconds": inference_seconds,
                    "env_seconds": time.perf_counter() - tic,
                    "active": int(active.sum()),
                    "strong": int(selected.sum()),
                    "cuda_peak_bytes": torch.cuda.max_memory_allocated(),
                }
            )
            if record_decisions:
                save(out / "decisions.json", decisions)
            save(out / "timing.json", costs)
            save(
                out / "progress.json",
                {
                    "time": time.time(),
                    "phase": "saved",
                    "query": query + 1,
                    "successes": int(success.sum()),
                },
            )
            del actions
        assert not np.any(~success & (counts < c["step_limit"]))
        episodes = [
            dict(
                **seeds[i],
                condition=c["condition"],
                success=bool(success[i]),
                submitted=int(counts[i]),
                frames=len(recorders[i].frames) if full_video else len(frames[i]),
                control_steps=recorders[i].ticks if full_video else None,
                frame_map=f"episode_{i:02d}_frames.json" if full_video else None,
                video=f"episode_{i:02d}.mp4" if mode != "none" else None,
                termination="success" if success[i] else "step_limit",
            )
            for i in range(batch)
        ]
        save(out / "episodes.json", episodes)
        if diagnose_q:
            q_evaluation.save(out)
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.venv.env_thread_pool.shutdown(wait=True)
        for recorder in recorders:
            recorder.close()
        for writer in videos:
            writer.release()
        if mode != "none":
            save(out / "frame_times.json", frames)
        if env is not None:
            env.offload(clear_cache=True)
        env = None
        gc.collect()
    save(
        out / "done.json",
        {
            "time": time.time(),
            "elapsed_seconds": time.time() - started,
            "episodes": batch,
            "successes": int(success.sum()),
            "condition": c["condition"],
            "takeover_mode": takeover,
            "record_full_video": full_video,
            "recording_mode": mode,
            "model_reused": reused,
            "unique_actual_seeds": len(set(actual)),
            "native_seed_retries": sum(s["native_retry"] for s in seeds),
            "strong_decisions": sum(d["strong"] for d in decisions),
            "decisions": len(decisions),
            "queries": len(costs),
            "teacher_shadow": shadow,
            "q_diagnostics": diagnose_q,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    try:
        run(args.config)
    except BaseException:
        c = read(args.config)
        save(
            Path(c["output"]) / "error.json",
            {"time": time.time(), "error": traceback.format_exc()},
        )
        raise
