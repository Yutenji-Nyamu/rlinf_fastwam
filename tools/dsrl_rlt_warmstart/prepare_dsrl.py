# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Prepare two source-locked clean DSRL slots without stopping predecessors."""

import copy
import json
import os
import time
from pathlib import Path

from omegaconf import OmegaConf

BASE = Path("/data/chenyiteng/deployment-20261106/dsrl-rlt-warmstart")
ROOT = Path(__file__).resolve().parents[2]
ACTIVE = Path("/data/chenyiteng/deployment-20261105/dsrl-fp32-accel/active-runs.json")
S = Path("/data/chenyiteng/deployment-20261008/bc-signal-tau-v1")


def read(p):
    return json.loads(Path(p).read_text())


def save(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2))


def main():
    assert os.getuid() == 1003
    active = read(ACTIVE)
    entries = {}
    for gpu, bench in [(4, "rt2"), (5, "dojo")]:
        old = Path(active["slots"][str(gpu)]["deployment"])
        oldowner = Path(active["slots"][str(gpu)]["owner"])
        op = read(old / "plan.json")
        oldroot = op["training_source"]
        oldrun = op["run"]
        d = BASE / f"{bench}-dsrl-clean-g{gpu}"
        d.mkdir(parents=True, exist_ok=False)
        run = Path("/data/chenyiteng/results/dsrl-rlt-warmstart-20261106") / d.name
        run.mkdir(parents=True, exist_ok=False)

        def relocate(x):
            return json.loads(
                json.dumps(x)
                .replace(oldroot, str(ROOT))
                .replace(str(old), str(d))
                .replace(oldrun, str(run))
            )

        p = relocate(op)
        predecessor = read(oldowner / "identity.json")
        p.update(
            training_source=str(ROOT),
            run=str(run),
            gpu=gpu,
            gpus=[gpu],
            bench=bench,
            namespace=f"RLinf_warmstart_dsrl_g{gpu}_20261106",
            pre_registered=True,
            predecessors=[predecessor],
            from_rlt=False,
            model_contract_required=False,
            acceleration_trial=False,
            protected_output=f"/data/chenyiteng/models/protected-eval/dsrl-rlt-warmstart-20261106/{d.name}",
        )
        p.pop("defer_formal_to", None)
        p["gpu_admission"] = read(S / "plan.json")["slots"][str(gpu)]
        p["phases"] = [
            {
                "name": "group_eval",
                "command": [
                    p["python"],
                    "-u",
                    "-B",
                    str(ROOT / "tools/dsrl_rlt_warmstart/group_eval.py"),
                    str(d),
                ],
                "timeout": 7200 if bench == "dojo" else 3600,
            }
        ]
        for phase in ("smoke", "smoke_resume", "formal"):
            p["phases"].append(
                {
                    "name": phase,
                    "command": [
                        p["python"],
                        "-u",
                        "-B",
                        str(ROOT / "tools/dojo_offpolicy/training_entry.py"),
                        "--config-path",
                        str(d / "config"),
                        "--config-name",
                        phase,
                    ],
                    "timeout": None if phase == "formal" else 2400,
                }
            )
        c = relocate(
            OmegaConf.to_container(
                OmegaConf.load(old / "config/formal.yaml"), resolve=True
            )
        )
        c["runner"].update(max_steps=400, resume_dir=None)
        c["runner"]["logger"].update(
            log_path=str(run / "formal"), experiment_name="formal"
        )
        c["algorithm"]["replay_buffer"]["warmup_size"] = 2000
        c["algorithm"].update(dsrl_critic_warmup_updates=2000, critic_actor_ratio=2)
        assert (
            c["algorithm"]["utd_ratio"] == 20
            and c["algorithm"]["backup_entropy"] is False
        )
        assert c["actor"]["global_batch_size"] == c["actor"]["micro_batch_size"] == 256
        c["actor"].update(defer_metrics=True, foreach_target_update=bench == "dojo")
        for opt in (
            c["actor"]["optim"],
            c["actor"]["critic_optim"],
            c["algorithm"]["entropy_tuning"]["optim"],
        ):
            opt["fused"] = True
        c["actor"]["model"]["openpi"].update(
            dsrl_noise_groups=50,
            dsrl_action_noise_dim=1600,
            dsrl_vectorized_q=True,
            dsrl_compile_q=bench == "rt2",
            dsrl_compile_q_mode="reduce-overhead",
            dsrl_u_enabled=False,
        )
        c["rollout"]["model"] = copy.deepcopy(c["actor"]["model"])
        (d / "config").mkdir()
        OmegaConf.save(OmegaConf.create(c), d / "config/formal.yaml")
        for phase, steps in [("smoke", 1), ("smoke_resume", 2)]:
            sc = copy.deepcopy(c)
            sc["runner"].update(
                max_steps=steps,
                val_check_interval=0,
                save_interval=1,
                resume_dir=None
                if phase == "smoke"
                else str(run / "smoke/smoke/checkpoints/global_step_1"),
            )
            sc["runner"]["logger"].update(
                log_path=str(run / phase), experiment_name=phase
            )
            sc["algorithm"].update(utd_ratio=2, dsrl_critic_warmup_updates=16)
            sc["algorithm"]["replay_buffer"]["warmup_size"] = 8
            horizon = 50 if bench == "dojo" else 40
            sc["env"]["train"].update(
                max_steps_per_rollout_epoch=horizon, max_episode_steps=horizon
            )
            OmegaConf.save(OmegaConf.create(sc), d / f"config/{phase}.yaml")
        e = relocate(read(old / "training-environment.json"))
        e.update(DOJO_GRPO_DEPLOYMENT=str(d), PYTHONDONTWRITEBYTECODE="1")
        save(d / "training-environment.json", e)
        if bench == "dojo":
            save(
                d / "sim-environment.json", relocate(read(old / "sim-environment.json"))
            )
            n = relocate(read(old / f"native-g{gpu}.json"))
            n["acceleration"] = {}
            save(d / f"native-g{gpu}.json", n)
            (d / "rpc").mkdir(mode=0o700)
            (d / "rpc/authkey").write_bytes(os.urandom(32))
            (d / "rpc/authkey").chmod(0o600)
        save(d / "plan.json", p)
        entries[str(gpu)] = {
            "deployment": str(d),
            "owner": str(d),
            "old_owner": str(oldowner),
            "predecessor": predecessor,
            "run": str(run),
            "bench": bench,
            "source": str(ROOT),
        }
    save(BASE / "prepared-dsrl.json", {"time": time.time(), "entries": entries})
    print(json.dumps({"prepared": entries}, indent=2))


if __name__ == "__main__":
    main()
