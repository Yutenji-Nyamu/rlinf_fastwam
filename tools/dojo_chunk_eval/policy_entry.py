# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Reuse the installed native JAX model with paired noise and bounded batch QA."""

import os
import sys
import time
from pathlib import Path

import numpy as np
from common import read, save


def install(model):
    import jax
    import jax.numpy as jnp
    from openpi.models import model as base
    from XPolicyLab.policy.Pi_05.model import slice_stacked_obs, unpack_robot_state

    policy = model.policy
    assert not policy._is_pytorch_model and policy._model.action_horizon == 50
    native_reset, native_update = model.reset, model.update_obs_batch
    modes = {}
    job = {}
    meta = []

    def reset():
        nonlocal job
        job = read(read(os.environ["DOJO_EVAL_POINTER"])["config"])
        assert job["num_steps"] in (1, 10) and job["execute_chunk"] in (10, 50)
        policy._sample_kwargs["num_steps"] = job["num_steps"]
        native_reset()

    def update(obs):
        nonlocal meta
        meta = [o["_eval_meta"] for o in obs]
        return native_update(obs)

    def noise_for(rows):
        return np.stack(
            [
                np.random.default_rng(
                    np.random.SeedSequence([job["noise_seed"], m["layout"], m["query"]])
                ).standard_normal((50, policy._model.action_dim), dtype=np.float32)
                for m in rows
            ]
        )

    def sequential(observations, noise):
        return np.stack(
            [policy.infer(o, noise=z)["actions"] for o, z in zip(observations, noise)]
        )

    def transformed_batch(observations, noise):
        # Keep native transforms per example; only the neural network is batched.
        rows = [
            policy._input_transform(jax.tree.map(lambda x: x, o)) for o in observations
        ]
        inputs = jax.tree.map(lambda *v: jnp.asarray(np.stack(v)), *rows)
        actions = np.asarray(
            policy._sample_actions(
                jax.random.key(0),
                base.Observation.from_dict(inputs),
                **dict(policy._sample_kwargs, noise=jnp.asarray(noise)),
            )
        )
        return np.stack(
            [
                policy._output_transform(
                    {"state": np.asarray(inputs["state"][i]), "actions": a}
                )["actions"]
                for i, a in enumerate(actions)
            ]
        )

    def native_batch(observations, noise):
        from XPolicyLab.policy.Pi_05.model import stack_obs

        return np.asarray(policy.infer(stack_obs(observations), noise=noise)["actions"])

    def actions(env_idx_list=None):
        ids = model._latest_env_idx_list if env_idx_list is None else env_idx_list
        assert list(ids) == list(model._latest_env_idx_list)
        observations = [
            slice_stacked_obs(model.observation_window, i) for i in range(len(ids))
        ]
        noise = noise_for(meta)
        # Pad active slots to the accepted eight-env shape to avoid recompilation.
        count = len(ids)
        assert 0 < count <= job["num_envs"] == 8
        observations += [observations[-1]] * (8 - count)
        noise = np.concatenate([noise, np.repeat(noise[-1:], 8 - count, axis=0)])
        n = job["num_steps"]
        if n not in modes:
            t = time.monotonic()
            reference = sequential(observations, noise)
            seq_seconds = time.monotonic() - t
            attempts = []
            modes[n] = "sequential"
            for name, fn in [
                ("native_batch", native_batch),
                ("transformed_batch", transformed_batch),
            ]:
                try:
                    t = time.monotonic()
                    candidate = fn(observations, noise)
                    cold = time.monotonic() - t
                    np.testing.assert_allclose(
                        candidate, reference, atol=2e-3, rtol=2e-3
                    )
                    t = time.monotonic()
                    fn(observations, noise)
                    warm = time.monotonic() - t
                    attempts.append(
                        {
                            "mode": name,
                            "passed": True,
                            "cold_seconds": cold,
                            "warm_seconds": warm,
                            "max_abs_difference": float(
                                np.max(np.abs(candidate - reference))
                            ),
                        }
                    )
                    modes[n] = name
                    break
                except Exception as exc:
                    attempts.append({"mode": name, "passed": False, "error": repr(exc)})
            save(
                Path(os.environ["DOJO_EVAL_DEPLOYMENT"])
                / f"batch-n{n}-acceptance.json",
                {
                    "mode": modes[n],
                    "sequential_cold_seconds": seq_seconds,
                    "attempts": attempts,
                    "horizon": 50,
                    "batch": 8,
                    "noise": "explicit paired per layout/query",
                },
            )
        started = time.monotonic()
        fn = {
            "native_batch": native_batch,
            "transformed_batch": transformed_batch,
            "sequential": sequential,
        }[modes[n]]
        result = fn(observations, noise)[:count]
        if job.get("probe_queries"):
            q = int(meta[0]["query"])
            np.savez_compressed(
                Path(job["output"]) / f"probe-input-q{q}.npz",
                state=model.observation_window["state"],
                actions=result,
                **model.observation_window["images"],
            )
        save(
            Path(job["output"]) / "policy-progress.json",
            {
                "time": time.time(),
                "phase": "policy_query",
                "queries": [m["query"] for m in meta],
                "num_steps": n,
                "batch_mode": modes[n],
                "active_envs": count,
                "seconds": time.monotonic() - started,
            },
        )
        return [
            unpack_robot_state(
                a, model.action_type, model.robot_action_dim_info, source_type="obs"
            )
            if model.robot_action_dim_info is not None
            else a
            for a in result
        ]

    model.reset = reset
    model.update_obs_batch = update
    model.get_action_batch = actions


def main():
    cfg = read(sys.argv[1])
    repo = Path(cfg["repo"])
    sys.path[:0] = [str(repo), str(repo / "XPolicyLab")]
    import XPolicyLab.setup_policy_server as server
    from XPolicyLab.policy.Pi_05.model import Model

    original = Model.__init__

    def initialize(self, model_cfg):
        original(self, model_cfg)
        install(self)

    Model.__init__ = initialize
    server.main(cfg["policy"])


if __name__ == "__main__":
    main()
