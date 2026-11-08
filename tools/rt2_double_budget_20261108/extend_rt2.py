# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Prepare/arm 200->400 and 313->626 without stopping either GPU trainer.

Queue transaction follows robotwin_dsrl_full/extend_four_to400.py; only the two
CPU schedulers are restarted. Existing trainers block fallback until they exit.
"""

import copy
import fcntl
import os
import signal
import sys
import time
from pathlib import Path

from continue_rt2 import ACTIVE, S, checkpoint_ready, read, same, save
from omegaconf import OmegaConf

B = Path("/data/chenyiteng/deployment-20261108/rt2-double-budget")
PAIR = Path("/data/chenyiteng/deployment-20261104/parallel-u47")
TARGETS = {"4": (200, 400), "7": (313, 626)}
PY = "/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python"


def prepare():
    assert os.getuid() == 1003 and not (B / "prepared.json").exists()
    active, scheduler = read(ACTIVE), read(S / "plan.json")
    entries = {}
    for g, (old_end, target) in TARGETS.items():
        old = Path(active["slots"][g]["owner"])
        predecessor = read(old / "identity.json")
        assert same(predecessor) and read(old / "status.json")["phase"] == "formal"
        assert scheduler["slots"][g]["external_identity"] == predecessor
        d = B / old.name
        (d / "config").mkdir(parents=True, exist_ok=False)
        p = read(old / "plan.json")
        assert p["bench"] == "rt2" and p["gpu"] == int(g)
        cfg = OmegaConf.to_container(
            OmegaConf.load(old / "config/formal.yaml"), resolve=True
        )
        assert (
            cfg["runner"]["max_steps"] == old_end
            and cfg["runner"]["max_epochs"] >= target
        )
        cp = (
            Path(cfg["runner"]["logger"]["log_path"])
            / cfg["runner"]["logger"]["experiment_name"]
            / "checkpoints"
            / f"global_step_{old_end}"
        )
        original = copy.deepcopy(cfg)
        cfg["runner"].update(max_steps=target, resume_dir=str(cp))
        check = copy.deepcopy(cfg)
        check["runner"].update(
            max_steps=old_end, resume_dir=original["runner"].get("resume_dir")
        )
        assert (
            check == original
        )  # All learning, data and evaluation settings stay identical.
        OmegaConf.save(OmegaConf.create(cfg), d / "config/formal.yaml")
        formal = copy.deepcopy(next(x for x in p["phases"] if x["name"] == "formal"))
        formal["command"] = [
            x.replace(str(old / "config"), str(d / "config")) for x in formal["command"]
        ]
        formal["timeout"] = None
        p.update(
            namespace=p["namespace"] + "_double",
            from_rlt=False,
            pre_registered=True,
            predecessors=[predecessor],
            phases=[formal],
            gpu_admission=copy.deepcopy(scheduler["slots"][g]),
        )
        save(d / "plan.json", p)
        env = read(old / "training-environment.json")
        for k, v in env.items():
            if isinstance(v, str):
                env[k] = v.replace(str(old), str(d))
        env["DOJO_GRPO_DEPLOYMENT"] = str(d)
        save(d / "training-environment.json", env)
        meta = {
            "gpu": int(g),
            "method": p["method"],
            "predecessor": predecessor,
            "previous_deployment": str(old),
            "previous_target": old_end,
            "checkpoint": str(cp),
            "deployment": str(d),
            "owner": str(d),
            "target": target,
            "source": p["training_source"],
            "run": p["run"],
            "episodes": target * 16,
        }
        save(d / "continuation.json", meta)
        entries[g] = meta
    save(B / "prepared.json", {"time": time.time(), "entries": entries})
    print(
        {
            g: {
                k: e[k] for k in ["previous_target", "target", "episodes", "checkpoint"]
            }
            for g, e in entries.items()
        }
    )


def check(g):
    """One production config validation per process; no GPU workload or replay load."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    e = read(B / "prepared.json")["entries"][g]
    d = Path(e["deployment"])
    sys.path.insert(0, e["source"])
    os.environ["PYTHONPATH"] = e["source"]
    from rlinf.config import validate_cfg
    from rlinf.scheduler import Cluster

    Cluster.NAMESPACE = "RLinf_double_cpu_" + str(os.getpid())
    c = validate_cfg(OmegaConf.load(d / "config/formal.yaml"))
    assert min(c.runner.max_steps, c.runner.max_epochs) == e["target"]
    root = Path(e["checkpoint"]).parent
    latest = max(
        root.glob("global_step_*"), key=lambda x: int(x.name.rsplit("_", 1)[-1])
    )
    assert checkpoint_ready(latest, e["method"]), str(latest)
    save(
        d / "config-check.json",
        {
            "result": "pass",
            "target": e["target"],
            "latest_existing_checkpoint": str(latest),
            "format": e["method"],
            "full_resume_files": True,
        },
    )
    print(g, "pass", e["target"], latest.name)


def arm():
    assert os.getuid() == 1003 and not (B / "armed.json").exists()
    entries = read(B / "prepared.json")["entries"]
    dsrl_root = entries["4"]["source"]
    sys.path.insert(0, dsrl_root)
    from common import exact_signal
    from tools.dojo_offpolicy import swap_4567 as swap

    def launch(d, args, root=dsrl_root, **kwargs):
        swap.R = Path(root)
        return swap.launch(Path(d), args, cwd=Path(root), **kwargs)

    def stop_cpu(who):
        exact_signal(who, signal.SIGTERM)
        deadline = time.monotonic() + 30
        while same(who):
            assert time.monotonic() < deadline
            time.sleep(0.1)

    active, sp, st = read(ACTIVE), read(S / "plan.json"), read(S / "status.json")
    jp = read(PAIR / "join-plan.json")
    join = read(PAIR / "join-identity.json")
    assert same(join) and read(PAIR / "join-status.json")["phase"] == "WAIT_BOTH_U"
    assert jp["jobs"] == [entries[g]["previous_deployment"] for g in ["4", "7"]]
    for g, e in entries.items():
        assert (
            same(e["predecessor"])
            and active["slots"][g]["owner"] == e["previous_deployment"]
        )
        assert (
            sp["slots"][g]["external_identity"] == e["predecessor"]
            and st["slots"][g]["phase"] == "WAIT_EXTERNAL"
        )
        assert read(Path(e["deployment"]) / "config-check.json")["result"] == "pass"
    owners = {
        g: launch(
            e["owner"],
            [PY, "-u", "-B", str(B / "package/continue_rt2.py"), e["owner"]],
            root=e["source"],
        )
        for g, e in entries.items()
    }
    save(B / "waiting-owners.json", owners)
    save(B / "previous-join-plan.json", jp)
    stop_cpu(join)
    scheduler_restarted = False
    try:
        with (S / "priority-handoff.lock").open("a") as serial:
            fcntl.flock(serial, fcntl.LOCK_EX)
            stop_cpu(read(S / "owner-identity.json"))
            try:
                with (S / "owner.lock").open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    p, state = read(S / "plan.json"), read(S / "status.json")
                    save(B / "previous-rlt-plan.json", p)
                    untouched = {g: copy.deepcopy(p["slots"][g]) for g in ["5", "6"]}
                    for g, e in entries.items():
                        assert (
                            same(e["predecessor"])
                            and p["slots"][g]["external_identity"] == e["predecessor"]
                        )
                        p["slots"][g].update(
                            external_identity=owners[g],
                            external_release=str(Path(e["owner"]) / "released.json"),
                            external_blockers=[e["predecessor"]],
                        )
                        state["slots"][g] = {"phase": "WAIT_EXTERNAL"}
                    assert all(p["slots"][g] == untouched[g] for g in untouched)
                    save(S / "plan.json", p)
                    save(S / "status.json", state)
                    active = read(ACTIVE)
                    for g, e in entries.items():
                        active["slots"][g].update(
                            total_target_steps=e["target"],
                            total_target_episodes=e["episodes"],
                            pending_continuation={
                                "owner": e["owner"],
                                "identity": owners[g],
                                "resume_step": e["previous_target"],
                                "target": e["target"],
                            },
                        )
                        save(
                            Path(e["owner"]) / "registered.json",
                            {"identity": owners[g], "time": time.time()},
                        )
                    active["updated"] = time.time()
                    save(ACTIVE, active)
            finally:
                launch(
                    B,
                    [
                        "/usr/bin/python3",
                        "-u",
                        "-B",
                        str(S / "ops/owner.py"),
                        "--plan",
                        str(S / "plan.json"),
                    ],
                    root=str(S),
                    name="rlt-owner.log",
                    identity=None,
                )
                scheduler_restarted = True
        jp["jobs"] = [entries[g]["owner"] for g in ["4", "7"]]
        save(PAIR / "join-plan.json", jp)
    finally:
        next_join = launch(
            PAIR,
            [
                PY,
                "-u",
                "-B",
                str(Path(dsrl_root) / "tools/robotwin_dsrl_full/join_fql.py"),
                str(PAIR),
            ],
            name="join-rt2-double.log",
            identity=None,
        )
        fp = Path(jp["fql"]) / "formal-plan.json"
        fplan = read(fp)
        fplan["predecessor_identity"] = next_join
        save(fp, fplan)
    assert scheduler_restarted and all(
        same(e["predecessor"]) and same(owners[g]) for g, e in entries.items()
    )
    for g, e in entries.items():
        launch(
            B,
            [
                PY,
                "-u",
                "-B",
                "-c",
                f"from tools.dojo_offpolicy.recover_queue import watch_owner; watch_owner({e['owner']!r})",
            ],
            root=e["source"],
            name=f"watch-g{g}.log",
            identity=None,
        )
    receipt = {
        "time": time.time(),
        "entries": entries,
        "owners": owners,
        "join": next_join,
        "current_trainers_untouched": True,
        "queue_waits_for_extended_runs": True,
        "untouched_slots": [5, 6],
    }
    save(B / "armed.json", receipt)
    print(receipt)


if __name__ == "__main__":
    {"prepare": prepare, "check": lambda: check(sys.argv[2]), "arm": arm}[sys.argv[1]]()
