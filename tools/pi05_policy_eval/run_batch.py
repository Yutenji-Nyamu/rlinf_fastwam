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
from protocol import choose, read, save, validate_config

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/pi05_signals"))
from signal_math import ugrow_10_vs_5  # noqa: E402


class EndpointOnly:
    """The accepted sampler can collect endpoints without attention hooks."""

    def step(self, *_):
        return nullcontext()


def arr(x):
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def infer(model, obs, steps, signals=False):
    with torch.inference_mode():
        return model.predict_action_batch(
            obs,
            mode="eval",
            compute_values=False,
            num_steps=steps,
            return_dvac_telemetry=True,
            record_signals=signals,
            consistency_steps=5,
            noise=None,
        )


def run(config):
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
    signals = policy in ("calibration", "high", "low", "random")
    need_strong = policy in ("strong", "high", "low", "random")
    weak = (
        None
        if policy == "strong"
        else get_model(OmegaConf.create(c["model"])).cuda().eval()
    )
    strong = None
    if weak is not None:
        weak._signal_observer = EndpointOnly()
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
    env, videos = None, []
    success, counts = np.zeros(batch, bool), np.zeros(batch, int)
    decisions, costs, frames = [], [], [[] for _ in range(batch)]
    threshold = read(c["threshold_path"]) if policy in ("high", "low") else {}
    try:
        # Accepted order: warm the model before creating any SAPIEN scenes.
        dummy = {
            "main_images": torch.zeros((batch, 224, 224, 3), dtype=torch.uint8),
            "wrist_images": torch.zeros((batch, 2, 224, 224, 3), dtype=torch.uint8),
            "extra_view_images": None,
            "states": torch.zeros((batch, 14)),
            "task_descriptions": ["move the pillbottle to the pad"] * batch,
        }
        for model in (weak, strong):
            if model is not None:
                action, result = infer(model, dummy, c["num_steps"])
                assert tuple(action.shape) == (batch, 50, 14)
                del action, result
        if policy == "calibration":
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
            del a, b, trace
        del dummy
        ec = OmegaConf.create(c["env"])
        assert not ec.auto_reset and ec.ignore_terminations
        env = RoboTwinEnv(ec, batch, 0, 1, None)
        obs, _ = env.reset()
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
        random.seed(c["noise_seed"])
        np.random.seed(c["noise_seed"])
        heads = arr(obs["main_images"]).astype(np.uint8)
        height, width = heads.shape[1:3]
        for slot in range(batch):
            writer = cv2.VideoWriter(
                str(out / f"episode_{slot:02d}.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                4,
                (width, height),
            )
            assert writer.isOpened()
            videos.append(writer)

        def snapshot(images):
            for slot, image in enumerate(images):
                videos[slot].write(
                    cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
                )
                frames[slot].append(time.time())

        snapshot(heads)
        chunk = c["execute_chunk"]
        for query in range((c["step_limit"] + chunk - 1) // chunk):
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
            if weak is not None:
                actions, result = infer(weak, obs, c["num_steps"], signals)
                if signals:
                    trace = result["dvac_telemetry"]
                    score = ugrow_10_vs_5(
                        arr(trace["final_model_action"]),
                        arr(trace["side_action_model"]),
                    )
                    assert score.valid.all()
                    u = score.score.mean(axis=1)
                    save(out / f"u_{query:03d}.json", score.score.tolist())
                del result
            selected = choose(
                policy,
                u,
                active,
                threshold,
                np.random.default_rng(c["gate_seed"] + query).random(batch),
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
            length = min(chunk, c["step_limit"] - query * chunk)
            save(
                out / "progress.json",
                {"time": time.time(), "phase": "environment", "query": query},
            )
            tic = time.perf_counter()
            obss, _, _, _, _ = env.chunk_step(action[:, :length])
            obs = obss[-1]
            counts = np.array([int(e.task.take_action_cnt) for e in env.venv.envs])
            after = np.array([bool(e.task.eval_success) for e in env.venv.envs])
            snapshot(arr(obs["main_images"]))
            for slot in np.flatnonzero(active):
                decisions.append(
                    {
                        "slot": int(slot),
                        "query": query,
                        "actual_seed": actual[slot],
                        "u": float(u[slot]) if u is not None else None,
                        "strong": bool(selected[slot]),
                        "start": int(before[slot]),
                        "end": int(counts[slot]),
                        "offered_actions": length,
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
                frames=len(frames[i]),
                video=f"episode_{i:02d}.mp4",
                termination="success" if success[i] else "step_limit",
            )
            for i in range(batch)
        ]
        save(out / "episodes.json", episodes)
    finally:
        for writer in videos:
            writer.release()
        save(out / "frame_times.json", frames)
        if env is not None:
            env.venv.env_thread_pool.shutdown(wait=True)
            env.offload(clear_cache=True)
    save(
        out / "done.json",
        {
            "time": time.time(),
            "elapsed_seconds": time.time() - started,
            "episodes": batch,
            "successes": int(success.sum()),
            "condition": c["condition"],
            "unique_actual_seeds": len(set(actual)),
            "native_seed_retries": sum(s["native_retry"] for s in seeds),
            "strong_decisions": sum(d["strong"] for d in decisions),
            "decisions": len(decisions),
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
