"""Replay native actions; change light at a frozen state, then restore it."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from .common import cuda_guard, write_json, write_npz


def physics_state(task):
    parts = []
    for actor in task.scene.get_all_actors():
        pose = actor.get_pose()
        parts.extend([pose.p, pose.q])
    for articulation in task.scene.get_all_articulations():
        pose = articulation.get_pose()
        parts.extend([pose.p, pose.q, articulation.get_qpos(), articulation.get_qvel()])
    return np.concatenate(parts).astype(np.float64)


def views(raw):
    return np.stack(
        [raw[k] for k in ["full_image", "left_wrist_image", "right_wrist_image"]]
    )


def main(args):
    torch = cuda_guard()
    sys.path.insert(0, args.model_source)
    from omegaconf import OmegaConf

    from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv

    root = Path(args.data)
    out = root / "grbc"
    out.mkdir(parents=True, exist_ok=True)
    pool = json.loads((root / "rt-pool/queries.json").read_text())
    nodes = []
    source_root = Path(pool[0]["batch"]).parent.parent
    for task_name in ["turn_switch", "stack_bowls_two"]:
        eps = []
        for row in pool:
            if row["task"] == task_name and row["seed"] not in [e["seed"] for e in eps]:
                eps.append(row)
        eps = eps[:4]
        assert len(eps) == 4
        cfg = json.loads(
            (source_root / "configs" / f"{task_name}_b0.json").read_text()
        )["env"]
        cfg.update(total_num_envs=4, seed=0, auto_reset=False, ignore_terminations=True)
        seeds = out / f"{task_name}_seeds.json"
        write_json(seeds, {task_name: {"success_seeds": [e["seed"] for e in eps]}})
        cfg["seeds_path"] = str(seeds)
        cfg["task_config"]["save_path"] = str(out / "native" / task_name)
        cfg["video_cfg"]["video_base_dir"] = str(out / "native" / task_name / "unused")
        assert not cfg["task_config"]["domain_randomization"]["random_light"]
        torch.empty(1, device="cuda")  # bind CUDA before native SAPIEN initialization
        env = RoboTwinEnv(OmegaConf.create(cfg), 4, 0, 1, None)
        try:
            env.reset()
            actual = [int(sub.task.ep_num) for sub in env.venv.envs]
            assert set(actual) == {e["seed"] for e in eps}, (
                "Reset retried; cannot silently relabel the seed"
            )
            byseed = {e["seed"]: e for e in eps}
            source = Path(eps[0]["batch"])
            desired = {
                seed: {
                    p["query"]
                    for p in pool
                    if p["task"] == task_name and p["seed"] == seed
                }
                for seed in actual
            }
            for q in range(max(max(x) for x in desired.values()) + 1):
                for slot, sub in enumerate(env.venv.envs):
                    seed = actual[slot]
                    if q not in desired[seed]:
                        continue
                    before = physics_state(sub.task)
                    ambient = np.asarray(sub.task.scene.ambient_light).copy()
                    cpu = torch.get_rng_state().clone()
                    cuda = torch.cuda.get_rng_state().clone()
                    rng = np.random.get_state()
                    baseline = sub.get_obs()
                    try:
                        sub.task.scene.set_ambient_light([0.18, 0.25, 0.65])
                        alternate = sub.get_obs()
                        np.testing.assert_array_equal(physics_state(sub.task), before)
                    finally:
                        sub.task.scene.set_ambient_light(ambient)
                        restored = sub.get_obs()
                    np.testing.assert_array_equal(physics_state(sub.task), before)
                    assert torch.equal(cpu, torch.get_rng_state()) and torch.equal(
                        cuda, torch.cuda.get_rng_state()
                    )
                    after_rng = np.random.get_state()
                    assert (
                        rng[0] == after_rng[0]
                        and np.array_equal(rng[1], after_rng[1])
                        and rng[2:] == after_rng[2:]
                    )
                    arrays = [views(baseline), views(alternate)]
                    pair = f"{task_name}_{seed}_q{q:03d}"
                    write_npz(
                        out / "pairs" / f"{pair}.npz",
                        images=np.stack(arrays),
                        state=np.asarray(baseline["state"]),
                        physics_state=before,
                        ambient=np.stack([ambient, [0.18, 0.25, 0.65]]),
                    )
                    state_hash = hashlib.sha256(before.tobytes()).hexdigest()
                    for appearance in range(2):
                        nodes.append(
                            dict(
                                pair=pair,
                                task=task_name,
                                seed=seed,
                                query=q,
                                appearance=appearance,
                                physical_state_sha256=state_hash,
                                physics_exact=True,
                                semantic_event="unlabelled temporal anchor",
                                raw=str(out / "pairs" / f"{pair}.npz"),
                                source_slot=byseed[seed]["slot"],
                                restoration_max_rgb_delta=int(
                                    np.max(
                                        np.abs(
                                            views(baseline).astype(int)
                                            - views(restored).astype(int)
                                        )
                                    )
                                ),
                                source_success=byseed[seed]["success"],
                            )
                        )
                    write_json(out / "nodes.json", nodes)
                with np.load(source / f"query_{q:03d}.npz", allow_pickle=False) as z:
                    actions = np.stack(
                        [
                            z["noise4_env_actions"][byseed[seed]["slot"], 0]
                            for seed in actual
                        ]
                    )
                env.chunk_step(torch.from_numpy(actions))
        finally:
            env.offload()
    assert len(nodes) == 64
    write_json(
        out / "capture-complete.json",
        dict(
            nodes=len(nodes),
            pairs=32,
            views=3,
            intervention="ambient light only",
            semantic_scope="same physical state across appearance; semantic phase labels require frame review",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model-source", required=True)
    main(p.parse_args())
