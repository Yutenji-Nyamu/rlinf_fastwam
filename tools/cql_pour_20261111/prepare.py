"""Prepare replacements without touching any running GPU owner."""

import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from omegaconf import OmegaConf

B = Path("/data/chenyiteng/deployment-20261111/cql-pour")
R = Path("/data/chenyiteng/projects/rlinf-shenzhen/exports/cql-pour-20261111")
S = Path("/data/chenyiteng/deployment-20261008/bc-signal-tau-v1")
active = json.loads(
    Path(
        "/data/chenyiteng/deployment-20261105/dsrl-fp32-accel/active-runs.json"
    ).read_text()
)
schedule = json.loads((S / "plan.json").read_text())


def read(p):
    return json.loads(p.read_text())


def save(p, x):
    p.write_text(json.dumps(x, indent=2) + "\n")


entries = {}
for g, method in [("4", "dsrl"), ("6", "rlt")]:
    old = Path(active["slots"][g]["deployment"])
    oldp = read(old / "plan.json")
    d = B / ("g" + g)
    run = Path("/home/chenyiteng/results/cql-pour-20261111") / ("g" + g)
    root = R / method
    (d / "config").mkdir(parents=True, exist_ok=False)
    run.mkdir(parents=True, exist_ok=False)
    replacements = [
        (oldp["training_source"], str(root)),
        (str(old), str(d)),
        (oldp["run"], str(run)),
        (oldp["namespace"], "RLinf_cql_pour_20261111_g" + g),
    ]

    def relocate(x, mapping=tuple(replacements)):
        if isinstance(x, dict):
            return {k: relocate(v, mapping) for k, v in x.items()}
        if isinstance(x, list):
            return [relocate(v, mapping) for v in x]
        if isinstance(x, str):
            for a, b in mapping:
                x = x.replace(a, b)
        return x

    p = relocate(oldp)
    p.update(
        pre_registered=True,
        predecessors=[active["slots"][g]["identity"]],
        from_rlt=False,
        gpu_admission=schedule["slots"][g],
        acceleration_trial=False,
        protected_output=str(
            Path("/home/chenyiteng/models/protected-eval/cql-pour-20261111") / ("g" + g)
        ),
    )
    for k in [
        "benchmark_replay_path",
        "old_deployment",
        "old_owner",
        "stage1_reuse",
        "probe_observation",
    ]:
        p.pop(k, None)
    c = relocate(
        OmegaConf.to_container(OmegaConf.load(old / "config/formal.yaml"), resolve=True)
    )
    command = lambda name, path, python=p["python"], root=root, d=d: [
        python,
        "-u",
        "-B",
        str(root / path),
        "--config-path",
        str(d / "config"),
        "--config-name",
        name,
    ]
    if g == "4":
        # Link a fully committed snapshot before old retention can unlink it.
        cps = Path(oldp["run"]) / "formal/formal/checkpoints"
        candidates = sorted(
            (
                x
                for x in cps.glob("global_step_*")
                if (x / "dojo-complete.json").is_file()
                and (x / "actor/local_shard_checkpoint/checkpoint_rank_0.pt").is_file()
            ),
            key=lambda x: int(x.name.split("_")[-1]),
        )
        cp = candidates[-1]
        pin = run / "resume" / cp.name
        shutil.copytree(cp, pin, copy_function=os.link)
        assert (pin / "actor/sac_components/replay_buffer/rank_0").is_dir()
        c["runner"]["resume_dir"] = str(pin)
        c["algorithm"].update(dsrl_cql_weight=0.5, dsrl_cql_temperature=1.0)
        p["phases"] = [
            {
                "name": "cql_smoke",
                "command": [
                    p["python"],
                    "-u",
                    "-B",
                    str(root / "tools/cql_pour_20261111/gpu_smoke.py"),
                    str(d),
                ],
                "timeout": 1200,
            },
            {
                "name": "formal",
                "command": command(
                    "formal", Path("tools/dojo_offpolicy/training_entry.py")
                ),
                "timeout": None,
            },
        ]
        save(
            d / "resume-source.json",
            {
                "source": str(cp),
                "pinned": str(pin),
                "time": time.time(),
                "contract": "complete checkpoint; model, optimizer, target, replay and counters retained",
            },
        )
    else:
        task = "pour_liquid_into_cup"
        p.update(
            task=task,
            stage1_data="/home/chenyiteng/datasets/robodojo-rlt-pour-20261111",
            reference_eval_episodes=32,
        )
        p["pre_native_phases"] = [
            {
                "name": "stage1",
                "command": [
                    p["python"],
                    "-u",
                    "-B",
                    str(root / "tools/dsrl_rlt_warmstart/stage1.py"),
                    str(d),
                ],
                "timeout": 18000,
            }
        ]
        p["phases"] = [
            {
                "name": "reference_eval",
                "command": [
                    p["python"],
                    "-u",
                    "-B",
                    str(root / "tools/dsrl_rlt_warmstart/reference_eval.py"),
                    str(d),
                ],
                "timeout": 3600,
            }
        ]
        for phase in ["smoke", "smoke_resume", "formal"]:
            p["phases"].append(
                {
                    "name": phase,
                    "command": command(
                        phase, Path("tools/dsrl_rlt_warmstart/rlt_training_entry.py")
                    ),
                    "timeout": 2400 if phase != "formal" else None,
                }
            )
        c["runner"].update(resume_dir=None, max_steps=79)
        token = str(run / "stage1/token-final.pt")
        c["rollout"]["rlt_feature_model"]["openpi"]["rlt_token_path"] = token
        c["algorithm"]["rlt_resume"]["contract"].update(
            stage1_token_path=token,
            stage1_token_sha256="pending_stage1",
            task_name=task,
        )
        for mode in ["train", "eval"]:
            c["env"][mode]["task_config"]["task_name"] = task
            c["env"][mode]["max_episode_steps"] = 400
            c["env"][mode]["max_steps_per_rollout_epoch"] = 400
        sf = copy.deepcopy(c["rollout"]["rlt_feature_model"])
        sf["openpi"]["rlt_token_path"] = None
        OmegaConf.save(OmegaConf.create(sf), d / "config/stage1.yaml")
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
            sc["algorithm"]["rlt_schedule"].update(
                warmup_min_size=8,
                warmup_post_collect_updates=16,
                max_updates_per_train_step=16,
            )
            sc["algorithm"]["rlt_resume"]["contract"].update(
                warmup_transitions=8, warmup_updates=16
            )
            sc["env"]["train"].update(
                max_steps_per_rollout_epoch=100, max_episode_steps=100
            )
            OmegaConf.save(OmegaConf.create(sc), d / f"config/{phase}.yaml")
        native = relocate(read(old / "native-g6.json"))
        native.update(task=task, invalid_layouts={"train": [], "eval": []})
        save(d / "native-g6.json", native)
        save(d / "sim-environment.json", relocate(read(old / "sim-environment.json")))
        (d / "rpc").mkdir(mode=0o700)
        (d / "rpc/authkey").write_bytes(os.urandom(32))
        (d / "rpc/authkey").chmod(0o600)
        sys.path.insert(0, str(root))
        from tools.dsrl_rlt_warmstart.stage1 import dataset_spec

        dr, info, tasks, train, val, sha = dataset_spec(p)
        assert len(train) == 50 and len(val) == 10
        assert not (
            {x["episode_index"] for x in train} & {x["episode_index"] for x in val}
        )
        frames = sum(x["length"] for x in train)
        save(
            d / "data-contract.json",
            {
                "train_episodes": 50,
                "validation_episodes": 10,
                "train_frames": frames,
                "prefix_cache_gib": (frames + 256) * 768 * 2048 * 2 / 2**30,
                "subset_sha256": sha,
                "dataset": p["stage1_data"],
                "task": task,
            },
        )
    OmegaConf.save(OmegaConf.create(c), d / "config/formal.yaml")
    save(
        d / "training-environment.json",
        relocate(read(old / "training-environment.json")),
    )
    save(d / "plan.json", p)
    entries[g] = {
        "deployment": str(d),
        "owner": str(d),
        "old_owner": str(old),
        "predecessor": active["slots"][g]["identity"],
        "run": str(run),
        "bench": p["bench"],
        "source": str(root),
        "method": method,
    }
save(B / "prepared-dsrl.json", {"time": time.time(), "entries": entries})
print(json.dumps(entries))
