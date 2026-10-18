"""Observe the installed official JAX Pi_05; keep its M10 infer path intact."""

import os
import sys
import time
from pathlib import Path

import numpy as np
from common import episode_dir, progress, read, save


class Observer:
    def __init__(self, model):
        import jax

        self.jax = jax
        self.model = model
        self.policy = model.policy
        assert not self.policy._is_pytorch_model
        assert self.policy._model.action_horizon == 50
        self.sample = self.policy._sample_actions
        self.original_infer = self.policy.infer
        self.policy._sample_actions = self.sample_observed
        self.policy.infer = self.infer
        self.meta = []
        self.index = 0
        self.accepted = False
        update = model.update_obs_batch
        actions = model.get_action_batch

        def update_obs(obs_list):
            self.meta = [dict(o["_u_meta"]) for o in obs_list]
            return update(obs_list)

        def get_actions(*args, **kwargs):
            self.index = 0
            result = actions(*args, **kwargs)
            assert self.index == len(self.meta)
            return result

        model.update_obs_batch = update_obs
        model.get_action_batch = get_actions

    def sample_observed(self, rng, observation, **kwargs):
        # The main call and its arguments are exactly the official call.
        assert kwargs.get("num_steps", 10) == 10
        main = self.sample(rng, observation, **kwargs)
        shape = (
            observation.state.shape[0],
            self.policy._model.action_horizon,
            self.policy._model.action_dim,
        )
        noise = kwargs.get("noise")
        if noise is None:
            noise = self.jax.random.normal(rng, shape)
        side = self.sample(rng, observation, **dict(kwargs, num_steps=5))
        a, b = np.asarray(main), np.asarray(side)
        assert a.shape == b.shape and a.shape[1] == 50 and a.shape[-1] >= 14
        # Reuse the accepted Robotwin mathematical definition, normalized D14.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pi05_signals"))
        from signal_math import ugrow_10_vs_5

        u = ugrow_10_vs_5(a[..., :14], b[..., :14])
        assert u.valid.all()
        self.trace = dict(
            main_endpoint=a[0],
            side_endpoint=b[0],
            initial_noise_full=np.asarray(noise)[0],
            u=u.score[0],
            u_valid=u.valid[0],
            sample_rng=self.jax.random.key_data(rng),
        )
        if not self.accepted:
            replay = self.sample(rng, observation, **kwargs)
            assert np.array_equal(a, np.asarray(replay)), "Main action parity failed"
            self.accepted = True
            save(
                Path(os.environ["DOJO_U_OUTPUT"]) / "main-parity.json",
                dict(
                    main_exact=True,
                    source="unchanged official JAX sampler",
                    main_steps=10,
                    side_steps=5,
                    horizon=50,
                    real_action_dims=14,
                ),
            )
        return main

    def infer(self, obs, **kwargs):
        meta = self.meta[self.index]
        self.index += 1
        out = episode_dir(meta)
        out.mkdir(parents=True, exist_ok=True)
        q = int(meta["action_before"]) // 50
        stem = f"query_{q:04d}"
        assert not (out / (stem + ".npz")).exists(), "Repeated query identity"
        before = self.policy._rng
        started = time.monotonic()
        result = self.original_infer(obs, **kwargs)
        expected = self.jax.random.split(before)[0]
        assert np.array_equal(
            self.jax.random.key_data(expected),
            self.jax.random.key_data(self.policy._rng),
        )
        self.trace.update(
            env_action=np.asarray(result["actions"]),
            rng_after=self.jax.random.key_data(self.policy._rng),
        )
        assert self.trace["env_action"].shape == (50, 14)
        np.savez_compressed(out / (stem + ".npz"), **self.trace)
        np.savez_compressed(
            out / (stem + "_obs.npz"), state=obs["state"], **obs["images"]
        )
        save(
            out / (stem + ".json"),
            dict(
                meta,
                query=q,
                prompt=obs["prompt"],
                seconds=time.monotonic() - started,
                rng_exact=True,
                u_coordinates="normalized model first 14 real coordinates",
                noise="same complete JAX initial noise, M10 vs complete M5",
            ),
        )
        progress("policy_query", layout=meta["layout"], query=q)
        return result


def main():
    cfg = read(sys.argv[1])
    repo = Path(cfg["repo"])
    sys.path[:0] = [str(repo), str(repo / "XPolicyLab")]
    import XPolicyLab.setup_policy_server as server
    from XPolicyLab.policy.Pi_05.model import Model

    original = Model.__init__

    def initialize(self, model_cfg):
        original(self, model_cfg)
        self._u_observer = Observer(self)

    Model.__init__ = initialize
    progress("policy_loading")
    server.main(cfg["policy"])


if __name__ == "__main__":
    main()
