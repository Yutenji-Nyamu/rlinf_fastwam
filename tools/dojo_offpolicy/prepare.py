# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Prepare reviewable single-card Dojo DSRL and GPU7 rendering gates, no launch."""

import copy
import json
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def prepare(root, previous, directory, template, gpu, num_envs=8):
    root, previous, d = map(Path, (root, previous, directory))
    d.mkdir(exist_ok=False)
    (d / "rpc").mkdir()
    (d / "rpc/authkey").write_bytes(os.urandom(32))
    (d / "rpc/authkey").chmod(0o600)
    p = json.loads((previous / "plan.json").read_text())
    p.update(
        training_source=str(root),
        gpu=gpu,
        gpus=[gpu],
        num_envs=num_envs,
        namespace=f"RLinf_dojo_offpolicy_{d.name}",
        run=f"/data/chenyiteng/results/dojo-offpolicy/{d.name}",
    )
    p["protected_output"] = (
        "/data/chenyiteng/models/protected-eval/dojo-dsrl-dustbin-20261103"
    )
    p["phases"] = []
    for name in ["smoke", "smoke_resume", "formal"] if gpu == 6 else ["render_probe"]:
        if name == "render_probe":
            command = [
                p["python"],
                "-u",
                "-B",
                str(root / "tools/dojo_offpolicy/render_probe.py"),
            ]
            timeout = 1000
        else:
            command = [
                p["python"],
                "-u",
                "-B",
                str(root / "tools/dojo_offpolicy/training_entry.py"),
                "--config-path",
                str(d / "config"),
                "--config-name",
                name,
            ]
            timeout = None if name == "formal" else 3600
        p["phases"].append({"name": name, "command": command, "timeout": timeout})
    save(d / "plan.json", p)
    e = json.loads((previous / "training-environment.json").read_text())
    for k in ("PYTHONPATH", "REPO_PATH", "RLINF_CODE_WORKING_DIR"):
        e[k] = str(root)
    e["EMBODIED_PATH"] = str(root / "examples/embodiment")
    e["DOJO_GRPO_DEPLOYMENT"] = str(d)
    save(d / "training-environment.json", e)
    save(
        d / "sim-environment.json",
        json.loads((previous / "sim-environment.json").read_text()),
    )
    n = json.loads((previous / "native-g6.json").read_text())
    n.update(
        gpu=gpu,
        num_envs=num_envs,
        output=str(d / f"native-g{gpu}"),
        socket=str(d / f"rpc/gpu{gpu}.sock"),
        authkey=str(d / "rpc/authkey"),
    )
    save(d / f"native-g{gpu}.json", n)
    c = OmegaConf.to_container(OmegaConf.load(template), resolve=True)
    old = OmegaConf.to_container(
        OmegaConf.load(previous / "config/formal.yaml"), resolve=True
    )
    c["env"] = copy.deepcopy(old["env"])
    for mode in ("train", "eval"):
        c["env"][mode].update(
            total_num_envs=num_envs,
            group_size=1,
            physical_gpus=[gpu],
            rpc_dir=str(d / "rpc"),
            use_rel_reward=False,
            reward_coef=1.0,
            enable_offload=False,
            rollout_epoch=1,
        )
    c["env"]["group_name"] = "EnvGroup"
    c["env"]["enable_offload"] = False
    assert 32 % num_envs == 0 and 800 % num_envs == 0
    c["env"]["eval"].update(rollout_epoch=32 // num_envs, fixed_layout_cycle=32)
    c["actor"]["model"].update(
        model_path=p["model"],
        num_action_chunks=50,
        num_steps=10,
        openpi_data=copy.deepcopy(old["actor"]["model"]["openpi_data"]),
    )
    c["actor"]["model"]["openpi"].update(
        config_name="pi05_robodojo", action_chunk=50, num_steps=10, action_horizon=50
    )
    c["actor"]["fsdp_config"].update(checkpoint_format="local_shard")
    c["actor"]["enable_offload"] = False
    c["rollout"]["enable_offload"] = False
    c["rollout"]["model"] = copy.deepcopy(c["actor"]["model"])
    c["cluster"] = {
        "num_nodes": 1,
        "component_placement": {"actor, env, rollout": str(gpu)},
    }
    c["runner"].update(
        max_steps=800 // num_envs,
        save_interval=32 // num_envs,
        val_check_interval=max(1, (800 // num_envs) // 8),
        resume_dir=None,
        weight_sync_interval=1,
        overlap_env_bootstrap=False,
        per_worker_log=False,
        per_worker_log_path=None,
    )
    c["runner"]["logger"].update(
        log_path=p["run"] + "/formal", experiment_name="formal"
    )
    (d / "config").mkdir()
    OmegaConf.save(OmegaConf.create(c), d / "config/formal.yaml")
    for phase, steps in (("smoke", 1), ("smoke_resume", 2)):
        s = copy.deepcopy(c)
        # Wiring smoke only: two real chunks, small UTD. Formal retains native
        # 700 actions and historical DSRL warmup/UTD. No smoke SR comparison.
        s["env"]["train"]["max_steps_per_rollout_epoch"] = 100
        s["runner"].update(max_steps=steps, val_check_interval=0, save_interval=1)
        s["runner"]["logger"].update(
            log_path=p["run"] + "/smoke", experiment_name="smoke"
        )
        s["algorithm"].update(utd_ratio=1)
        s["algorithm"]["replay_buffer"].update(warmup_size=num_envs)
        if phase == "smoke_resume":
            s["runner"]["resume_dir"] = (
                p["run"] + "/smoke/smoke/checkpoints/global_step_1"
            )
        OmegaConf.save(OmegaConf.create(s), d / f"config/{phase}.yaml")
    return p


if __name__ == "__main__":
    prepare(
        *sys.argv[1:5], int(sys.argv[5]), int(sys.argv[6]) if len(sys.argv) > 6 else 8
    )
