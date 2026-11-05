# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Prepare four fresh FP32/+-2 jobs. Does not stop or reserve GPUs."""

import copy
import json
import os
import time
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(
    "/data/chenyiteng/projects/rlinf-shenzhen/worktrees/dsrl-fp32-accel-20261105"
)
BASE = Path("/data/chenyiteng/deployment-20261105/dsrl-fp32-accel")
OLD = Path("/data/chenyiteng/deployment-20261105/dsrl400")
RESULTS = Path("/data/chenyiteng/results/dsrl-fp32-accel-20261105")
S = Path("/data/chenyiteng/deployment-20261008/bc-signal-tau-v1")


def read(p):
    return json.loads(Path(p).read_text())


def save(p, x):
    Path(p).write_text(json.dumps(x, indent=2) + "\n")


def main():
    assert os.getuid() == 1003
    entries = {}
    for gpu, name in [
        (4, "rt2-dsrl-clean-g4"),
        (5, "dojo-dsrl-full-g5"),
        (6, "dojo-dsrl-u-g6"),
        (7, "rt2-dsrl-u-g7"),
    ]:
        old = OLD / name
        d = BASE / name
        dojo = gpu in (5, 6)
        u = gpu in (6, 7)
        op = read(old / "plan.json")
        oldroot = op["training_source"]
        oldrun = op["run"]
        run = RESULTS / name
        d.mkdir(exist_ok=False)
        (d / "config").mkdir()
        run.mkdir(parents=True, exist_ok=False)

        def relocate(x):
            text = (
                json.dumps(x)
                .replace(oldroot, str(ROOT))
                .replace(str(old), str(d))
                .replace(oldrun, str(run))
            )
            return json.loads(text)

        plan = relocate(op)
        oldowner = old if dojo else old / "owner"
        owner = d if dojo else d / "owner"
        owner.mkdir(exist_ok=True)
        predecessors = [read(oldowner / "identity.json")]
        # The queued Dojo continuation still has a live original predecessor.
        continuation = read(oldowner / "continuation.json")
        if dojo:
            predecessors.append(continuation["predecessor"])
        plan.update(
            training_source=str(ROOT),
            run=str(run),
            namespace=f"RLinf_dsrl_fp32_g{gpu}_20261105",
            gpu=gpu,
            bench="dojo" if dojo else "rt2",
            python=op["python"],
            model_contract_required=False,
            pre_registered=True,
            predecessors=predecessors,
            acceleration_trial=u,
        )
        plan.pop("defer_formal_to", None)
        plan["protected_output"] = (
            f"/data/chenyiteng/models/protected-eval/dsrl-fp32-accel-20261105/{name}"
        )
        cfg = relocate(
            OmegaConf.to_container(
                OmegaConf.load(old / "config/formal.yaml"), resolve=True
            )
        )
        cfg["runner"].update(max_steps=400, resume_dir=None)
        cfg["actor"]["model"]["openpi"].update(
            dsrl_fp32_small=True,
            dsrl_noise_bound=2.0,
            dsrl_vectorized_q=False,
            dsrl_compile_q=False,
        )
        cfg["actor"]["defer_metrics"] = False
        cfg["actor"]["foreach_target_update"] = False
        for optim in (
            cfg["actor"]["optim"],
            cfg["actor"]["critic_optim"],
            cfg["algorithm"]["entropy_tuning"]["optim"],
        ):
            optim["fused"] = False
        assert cfg["actor"]["fsdp_config"]["mixed_precision"] == {
            "param_dtype": None,
            "reduce_dtype": None,
            "buffer_dtype": None,
        }
        assert (
            cfg["actor"]["global_batch_size"] == cfg["actor"]["micro_batch_size"] == 256
        )
        assert (
            cfg["algorithm"]["utd_ratio"] == 20
            and cfg["actor"]["model"]["openpi"]["dsrl_full_noise"]
        )
        OmegaConf.save(OmegaConf.create(cfg), d / "config/formal.yaml")
        for phase, steps, resume in [
            ("smoke", 1, None),
            ("smoke_resume", 2, str(run / "smoke/smoke/checkpoints/global_step_1")),
        ]:
            c = copy.deepcopy(cfg)
            c["runner"].update(
                max_steps=steps,
                val_check_interval=0,
                save_interval=1,
                resume_dir=resume,
            )
            c["runner"]["logger"].update(
                log_path=str(run / phase), experiment_name=phase
            )
            c["algorithm"]["utd_ratio"] = 2
            c["algorithm"]["replay_buffer"]["warmup_size"] = 8
            c["env"]["train"]["max_steps_per_rollout_epoch"] = 50 if dojo else 40
            c["env"]["train"]["max_episode_steps"] = 50 if dojo else 40
            OmegaConf.save(OmegaConf.create(c), d / f"config/{phase}.yaml")
        env = relocate(read(old / "training-environment.json"))
        env["DOJO_GRPO_DEPLOYMENT"] = str(d)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        save(d / "training-environment.json", env)
        if dojo:
            save(
                d / "sim-environment.json", relocate(read(old / "sim-environment.json"))
            )
            native = relocate(read(old / f"native-g{gpu}.json"))
            # Preserve already accepted rejected-layout bookkeeping, without
            # changing seed or introducing a new reset policy.
            previous = Path(continuation["previous_deployment"])
            invalid = {k: set(v) for k, v in native.get("invalid_layouts", {}).items()}
            log = previous / f"native-g{gpu}/invalid-resets.jsonl"
            for line in log.read_text().splitlines() if log.exists() else []:
                row = json.loads(line)
                invalid.setdefault(row["mode"], set()).update(row["invalid_layouts"])
            native["invalid_layouts"] = {k: sorted(v) for k, v in invalid.items()}
            save(d / f"native-g{gpu}.json", native)
            (d / "rpc").mkdir(mode=0o700)
            (d / "rpc/authkey").write_bytes(os.urandom(32))
            (d / "rpc/authkey").chmod(0o600)
            plan["phases"] = [
                {
                    "name": phase,
                    "command": [
                        plan["python"],
                        "-u",
                        "-B",
                        str(ROOT / "tools/dojo_offpolicy/training_entry.py"),
                        "--config-path",
                        str(d / "config"),
                        "--config-name",
                        phase,
                    ],
                    "timeout": None if phase == "formal" else 1800,
                }
                for phase in ("smoke", "smoke_resume", "formal")
            ]
        else:
            queue = read(oldowner / "plan.json")
            queue.update(
                training_source=str(ROOT),
                gpu=gpu,
                python=plan["python"],
                bench="rt2",
                acceleration_trial=u,
                deployment=str(d),
                from_rlt=False,
                predecessors=predecessors,
                pre_registered=True,
            )
            queue["phases"] = [
                {
                    "role": "u" if u else "clean",
                    "phase": phase,
                    "deployment": str(d),
                    "timeout": None if phase == "formal" else 1800,
                }
                for phase in ("smoke", "smoke_resume", "formal")
            ]
            if gpu == 4:
                queue["gpu_admission"] = read(S / "plan.json")["slots"]["4"]
            save(owner / "plan.json", queue)
        save(d / "plan.json", plan)
        entries[str(gpu)] = {
            "owner": str(owner),
            "deployment": str(d),
            "predecessors": predecessors,
            "old_owner": str(oldowner),
            "bench": plan["bench"],
        }
    save(
        BASE / "prepared-fp32.json",
        {"time": time.time(), "entries": entries, "source": str(ROOT)},
    )
    print(json.dumps({"prepared": entries, "active_tasks_untouched": True}))


if __name__ == "__main__":
    main()
