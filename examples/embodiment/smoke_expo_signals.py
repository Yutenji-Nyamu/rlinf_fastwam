"""Bounded native U/Norm smoke: one collection, two Q20/FM/editor calls.

Two real online windows are reserved per smoke batch to exercise non-unit
weights reliably. Production replay stays uniformly sampled. No formal launch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from train_expo_formal import native_boundary, restore_rng, rng_state
from train_expo_ft import check_finite, digest, frozen_sample, log

from rlinf.algorithms.expo_ft.backend import (
    Pi05Backend,
    clone_env_observation,
    create_robotwin_env,
)
from rlinf.algorithms.expo_ft.core import ExpoConfig, ExpoLearner
from rlinf.algorithms.expo_ft.formal_replay import FormalReplay
from rlinf.algorithms.expo_ft.lifecycle import close_robotwin_env
from rlinf.algorithms.expo_ft.signals import TRACE_FIELDS, SignalWeighting


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--max-episodes", type=int, default=3, choices=(1, 2, 3))
    args = parser.parse_args()
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=False)
    inputs = json.loads(Path(args.inputs).read_text())
    source = Path(__file__).resolve().parents[2]
    for name, expected in inputs["port_source_manifest"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Smoke source fingerprint differs: " + name)
    if inputs["physical_gpus"] != [6, 7] or inputs["formal"]["parallel_devices"] != 2:
        raise ValueError("This smoke requires the authorized SZ3 two-card recipe")
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    if torch.cuda.device_count() != 2:
        raise RuntimeError("Expected exactly two visible cards")
    for i in range(2):
        with torch.cuda.device(i):
            torch.cuda.manual_seed(seed + 1009 * i)
    log(
        run,
        "starting",
        physical_gpus=[6, 7],
        visible=os.environ["CUDA_VISIBLE_DEVICES"],
        max_episodes=args.max_episodes,
        batch_size=64,
        critic_updates=20,
        coverage_rows=2,
        note="Smoke only; no warmup/budget/evaluation claim; two real online rows per coverage batch",
    )
    cfg = inputs["formal"]
    backend = Pi05Backend(
        OmegaConf.create(inputs["model"]),
        parallel_devices=2,
        candidate_microbatch=8,
        observation_microbatch=64,
        fm_microbatch=64,
        source_head=inputs["source_head"],
        lr=cfg["base_lr"],
    )
    learner = ExpoLearner(ExpoConfig(**inputs["core"]), device="cuda:0", seed=seed)
    generator = torch.Generator(device="cuda:0").manual_seed(seed)
    methods = {}
    pools = {}
    for name, kind in (("u", "ugrow_10_5"), ("norm", "norm_residual_t5_l3")):
        settings = dict(inputs["signal"], kind=kind, target="both")
        methods[name] = SignalWeighting(settings)
        pools[name] = FormalReplay(
            run / ("replay-" + name),
            inputs["demo_path"],
            seed=seed,
            signal_contract=methods[name].contract,
        )
    frozen = frozen_sample(backend)
    env = None
    episodes = []
    try:
        for i in range(2):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
        env = create_robotwin_env(
            OmegaConf.create(inputs["env"]), num_envs=1, seed_offset=0
        )
        log(run, "renderer", **env.expo_renderer_binding)
        for episode in range(args.max_episodes):
            env_seed = int(env.success_seeds[episode % env.success_seeds.numel()])
            obs, _ = env.reset(env_seeds=[env_seed])
            frames = []
            actions = []
            rewards = []
            terms = []
            truncs = []
            traces = {name: {key: [] for key in TRACE_FIELDS} for name in methods}
            done = success = False
            query = 0
            while not done:
                # Observe Norm on original ODE10, then U only for the chosen parent.
                collected = backend.sample_normalized(
                    obs, 8, generator=generator, signal_kind="norm_residual_t5_l3"
                )
                base = collected["actions"][:, :, :10, :14]
                selected = learner.select_actions(backend.critic_observation(obs), base)
                norm, parent = backend.selected_signal(
                    obs, collected, selected["index"]
                )
                before = rng_state(generator)
                u, _ = backend.selected_signal(
                    obs, dict(collected, kind="ugrow_10_5"), selected["index"]
                )
                after = rng_state(generator)
                if digest(before) != digest(after):
                    raise AssertionError("U comparison changed main RNG")
                if episode == 0 and query == 0:
                    # Fixed-noise comparison checks producer transparency once.
                    processed = backend._prepare(obs)
                    saved = rng_state(generator)
                    with torch.inference_mode():
                        raw_plain = backend.parallel_adapter(
                            processed, collected["noise"][:, int(parent[0])], "sample"
                        )
                        raw_norm, norm_check = backend.parallel_adapter(
                            processed,
                            collected["noise"][:, int(parent[0])],
                            "sample_norm",
                        )
                    restore_rng(saved, generator)
                    torch.testing.assert_close(raw_plain, raw_norm, rtol=0, atol=0)
                    log(
                        run,
                        "producer_identity",
                        main_actions_equal=True,
                        u_rng_equal=True,
                        norm_min=float(norm.min()),
                        norm_max=float(norm.max()),
                        u_min=float(u.min()),
                        u_max=float(u.max()),
                    )
                signals = {"u": u[0].cpu(), "norm": norm[0].cpu()}
                canonical = backend.decode(obs, selected["actions"])
                for pos in range(10):
                    frames.append(clone_env_observation(obs))
                    command = canonical[:, pos : pos + 1].clone()
                    obs, reward, term, trunc, _ = env.step(command, auto_reset=False)
                    terminal, timeout, _ = native_boundary(
                        torch.as_tensor(term).any(), torch.as_tensor(trunc).any()
                    )
                    actions.append(command[0, 0].cpu())
                    rewards.append(float(torch.as_tensor(reward).reshape(-1)[0]))
                    terms.append(terminal)
                    truncs.append(timeout)
                    for name in methods:
                        row = dict(
                            raw=float(signals[name][pos]),
                            query=query,
                            position=pos,
                            parent=int(parent[0]),
                            edited=int(selected["index"][0]) >= 8,
                            base_version=collected["base_version"],
                        )
                        for key, value in row.items():
                            traces[name][key].append(value)
                    done = terminal or timeout
                    success = success or terminal
                    if done:
                        break
                    if len(actions) >= 200:
                        raise ValueError("Native episode cap missing")
                query += 1
                log(
                    run,
                    "chunk",
                    episode=episode,
                    query=query,
                    physical_actions=len(actions),
                    success=success,
                )
            for name, pool in pools.items():
                pool.append_episode(
                    frames,
                    torch.stack(actions),
                    rewards,
                    terms,
                    truncs,
                    success,
                    f"online-{episode:06d}",
                    clone_env_observation(obs),
                    signal_trace=traces[name],
                )
            episodes.append(
                dict(
                    episode=episode,
                    seed=env_seed,
                    actions=len(actions),
                    success=success,
                )
            )
            log(run, "episode", **episodes[-1])
            if success and len(actions) >= 50:
                break
        close_robotwin_env(env)
        env = None
        for i in range(2):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
        if not any(row["success"] and row["actions"] >= 50 for row in episodes):
            raise RuntimeError(
                "Bounded collection had no successful real H50; do not fabricate FM eligibility"
            )
        results = {}
        for name, pool in pools.items():
            method = methods[name]
            online_index = next(
                i
                for i, e in enumerate(pool.entries)
                if e["kind"] == "online" and e["fm_count"] > 0
            )
            real = pool.entries[online_index]
            draw = pool._draw

            def coverage_draw(size, fm=False):
                refs = draw(size, fm=fm)
                count = real["fm_count" if fm else "q_count"]
                refs[:2] = [(online_index, 0), (online_index, min(1, count - 1))]
                return refs

            pool._draw = coverage_draw

            def sampled():
                return pool.sample(64, backend, "cuda:0")

            def next_candidates(obs):
                return backend.sample_normalized(
                    obs["env_obs"], 8, generator=generator
                )[:, :, :10, :14]

            def fm():
                obs, actions, counts = pool.sample_fm(64)
                weights, metrics = method.weights(
                    pool.last_fm_signal,
                    "fm",
                    completed_episodes=len(episodes),
                    update_step=learner.update_calls,
                )
                if metrics["final_weight_std"] <= 0:
                    raise AssertionError("FM coverage weights remained neutral")
                out = backend.fm_update(obs, actions, action_weights=weights)
                out.update({f"signal/{k}": v for k, v in metrics.items()})
                log(run, "fm", variant=name, sources=counts, metrics=out)
                return out

            def editor_weights(batch):
                w, m = method.weights(
                    batch,
                    "editor",
                    completed_episodes=len(episodes),
                    update_step=learner.update_calls,
                )
                if m["final_weight_std"] <= 0:
                    raise AssertionError("Editor coverage weights remained neutral")
                return w, m

            started = time.monotonic()
            results[name] = learner.update_call(
                sampled, next_candidates, fm, editor_weights
            )
            results[name]["wall_seconds"] = time.monotonic() - started
            check_finite(results[name])
            log(
                run,
                "updated",
                variant=name,
                metrics=results[name],
                cache=pool.cache_metrics(),
            )
        if frozen_sample(backend) != frozen:
            raise AssertionError("Frozen prefix changed")
        # One complete save; exact in-process restoration checks changed optimizer/counters.
        saved = dict(
            base=backend.state_dict(),
            core=learner.state_dict(),
            replay={name: pool.state_dict() for name, pool in pools.items()},
            signal={name: method.contract for name, method in methods.items()},
            rng=rng_state(generator),
        )
        check_finite(saved)
        temp = run / "checkpoint.partial"
        torch.save(saved, temp)
        temp.replace(run / "checkpoint.pt")
        saved_calls = learner.update_calls
        learner.update_calls = -1
        backend.base_updates = -1
        backend.load_state_dict(saved["base"])
        learner.load_state_dict(saved["core"])
        for name, pool in pools.items():
            pool.load_state_dict(saved["replay"][name])
        restore_rng(saved["rng"], generator)
        if learner.update_calls != saved_calls or backend.base_updates != 2:
            raise AssertionError("Restore counters differ")
        result = dict(
            ok=True,
            episodes=episodes,
            results=results,
            checkpoint=str(run / "checkpoint.pt"),
            restored_calls=learner.update_calls,
            base_updates=backend.base_updates,
            critic_steps=learner.critic_steps,
            frozen_prefix_unchanged=True,
            full_save=True,
            restore="same-process state load; not a new-process resume",
            formal_enabled=False,
            coverage_rows=2,
            physical_gpus=[6, 7],
        )
        (run / "complete.json").write_text(json.dumps(result, indent=2))
        log(run, "complete", **result)
    except BaseException as exc:
        (run / "failure.json").write_text(
            json.dumps(dict(type=type(exc).__name__, error=str(exc)))
        )
        raise
    finally:
        if env is not None:
            close_robotwin_env(env)


if __name__ == "__main__":
    main()
