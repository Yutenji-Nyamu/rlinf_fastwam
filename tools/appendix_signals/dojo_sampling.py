"""Observe K4 next to frozen Dojo inference; leave main actions and RNG intact."""

import argparse
import copy
import inspect
import json
import time
from pathlib import Path

import numpy as np

from .common import Timer, cuda_guard, entropy, uniform_indices, write_json, write_npz


def add_legacy_trace(model):
    """Expose existing native outputs without changing the deployed sampler."""
    original = model.predict_action_batch
    if "return_dvac_telemetry" in inspect.signature(original).parameters:
        return
    sampler = model.sample_actions

    def predict(obs, *, noise=None, num_steps=None, **kwargs):
        if num_steps is not None:
            assert num_steps == model.config.num_steps
        if noise is not None:

            def with_noise(observation, *args, **options):
                options["noise"] = noise
                return sampler(observation, *args, **options)

            model.sample_actions = with_noise
        try:
            actions, result = original(obs, **kwargs)
        finally:
            model.sample_actions = sampler
        chains = result["forward_inputs"]["chains"]
        result["dvac_telemetry"] = {
            "initial_noise_full": chains[:, 0].detach(),
            "final_model_action": result["model_actions"][
                ..., : model.config.action_env_dim
            ].detach(),
        }
        return actions, result

    model.predict_action_batch = predict


def main(args):
    torch = cuda_guard()
    from omegaconf import OmegaConf

    from rlinf.envs import get_env_cls
    from rlinf.models.embodiment.openpi import get_model
    from tools.pi05_u_video.noise_samples import collect_noise_samples, numpy_exact

    d = Path(args.deployment)
    plan = json.loads((d / "plan.json").read_text())
    out = Path(args.data) / "dojo" / plan["task"]
    out.mkdir(parents=True, exist_ok=True)
    c = OmegaConf.load(d / "config/formal.yaml")
    mc = copy.deepcopy(c.rollout.get("rlt_feature_model", c.actor.model))
    mc.openpi.use_dsrl = False
    if "use_rlt" in mc.openpi:
        mc.openpi.use_rlt = False
    model = get_model(mc).cuda().eval()
    model.requires_grad_(False)
    add_legacy_trace(model)
    ec = copy.deepcopy(c.env.eval)
    ec.total_num_envs = 12
    ec.auto_reset = False
    ec.ignore_terminations = False
    env = get_env_cls(ec.env_type, ec)(ec, 12, 0, 1, None)
    start = time.monotonic()
    obs, _ = env.reset()
    reset_seconds = time.monotonic() - start
    chunk = int(mc.num_action_chunks)
    steps = int(ec.max_episode_steps)
    anchors = set(uniform_indices((steps + chunk - 1) // chunk, 12).tolist())
    done = np.zeros(12, bool)
    success = np.zeros(12, bool)
    records = []
    frames = []
    timings = []
    torch.manual_seed(20261111)
    torch.cuda.manual_seed(20261111)
    try:
        for q in range((steps + chunk - 1) // chunk):
            before = env.elapsed_steps.numpy().copy()
            active = ~done
            obs.setdefault("extra_view_images", None)
            frozen = {
                k: (
                    v.detach().cpu().numpy().copy()
                    if torch.is_tensor(v)
                    else np.asarray(v)
                )
                for k, v in obs.items()
                if v is not None
            }
            with Timer() as main_timer, torch.inference_mode():
                actions, r = model.predict_action_batch(
                    obs,
                    mode="eval",
                    compute_values=False,
                    num_steps=10,
                    return_dvac_telemetry=True,
                    record_signals=False,
                )
            trace = r["dvac_telemetry"]
            observed = None
            cpu = torch.get_rng_state().clone()
            cuda = torch.cuda.get_rng_state().clone()
            main_action = actions.clone()
            if q in anchors:
                with Timer() as extra_timer:
                    observed = collect_noise_samples(
                        model, obs, trace, actions, 202611110 + q, max_extra_batch=36
                    )
                assert torch.equal(main_action, actions)
                assert torch.equal(cpu, torch.get_rng_state()) and torch.equal(
                    cuda, torch.cuda.get_rng_state()
                )
                np.testing.assert_array_equal(
                    observed["noise4_env_actions"][:, 0], numpy_exact(actions)
                )
            start = time.monotonic()
            nxt, rewards, terms, truncs, info = env.chunk_step(actions)
            env_seconds = time.monotonic() - start
            after = env.elapsed_steps.numpy().copy()
            term = np.asarray(terms).reshape(12, -1).any(-1)
            trunc = np.asarray(truncs).reshape(12, -1).any(-1)
            success |= np.asarray(info[-1]["success"], dtype=bool) & active
            done |= term | trunc
            assert np.all((after - before) >= 0) and np.all((after - before) <= chunk)
            write_npz(
                out / "observations" / f"q{q:03d}.npz",
                **frozen,
                before=before,
                after=after,
                active=active,
                success=success.copy(),
            )
            write_npz(
                out / "frames" / f"after_q{q:03d}.npz",
                main_images=numpy_exact(nxt[-1]["main_images"]),
                native_action_steps=after,
                active_before=active,
                done=done.copy(),
                success=success.copy(),
            )
            if observed is not None:
                write_npz(
                    out / "queries" / f"q{q:03d}.npz",
                    **observed,
                    entropy=entropy(observed["noise4_model_actions"]),
                    before=before,
                    after=after,
                    active=active,
                    success=success.copy(),
                )
                records.append(
                    dict(
                        query=q,
                        path=f"queries/q{q:03d}.npz",
                        K=4,
                        H=int(observed["noise4_model_actions"].shape[2]),
                        C=int(actions.shape[1]),
                        executed_prefix=(after - before).tolist(),
                        rng_exact=True,
                        main_exact=True,
                    )
                )
            frames.append(
                dict(
                    query=q,
                    path=f"observations/q{q:03d}.npz",
                    native_action_steps=before.tolist(),
                    active=active.tolist(),
                )
            )
            timings.append(
                dict(
                    query=q,
                    main=main_timer.result,
                    extras=extra_timer.result if observed is not None else None,
                    environment_seconds=env_seconds,
                )
            )
            write_json(out / "queries.json", records)
            write_json(out / "frame_map.json", frames)
            write_json(out / "timings.json", timings)
            write_json(
                out / "progress.json",
                dict(query=q, successes=int(success.sum()), ended=int(done.sum())),
            )
            obs = nxt[-1]
            if done.all():
                break
        write_json(
            out / "episodes.json",
            [
                dict(
                    slot=i,
                    layout_id=int(env.reset_state_ids[i]),
                    success=bool(success[i]),
                    complete=bool(done[i]),
                    steps=int(env.elapsed_steps[i]),
                    task=plan["task"],
                )
                for i in range(12)
            ],
        )
        write_json(
            out / "complete.json",
            dict(
                status="complete",
                episodes=12,
                successes=int(success.sum()),
                frozen_reference=True,
                reset_seconds=reset_seconds,
                chunk=chunk,
                H=int(trace["final_model_action"].shape[1]),
                time_axis="native action steps; observation frames, not full-rate physics video",
                model=OmegaConf.to_container(mc, resolve=True),
            ),
        )
    finally:
        env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--deployment", required=True)
    p.add_argument("--data", required=True)
    main(p.parse_args())
