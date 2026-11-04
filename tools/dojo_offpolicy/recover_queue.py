# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Single-card adaptation of the accepted Dojo bounded recovery wrapper."""

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

from tools.dojo_grpo.recover_queue import read, simulation_failure


def abnormal_empty(now, phase_started, last_progress, empty_since):
    # Allow the observed model/Isaac startup. Resident training should never
    # lose every GPU context for two minutes without any log/progress activity.
    return (
        empty_since is not None
        and now - phase_started >= 600
        and now - empty_since >= 120
        and now - last_progress >= 120
    )


def release_missing(directory):
    """Hard-killed owner: exact cleanup and the same normal RLT release marker."""
    from tools.dojo_offpolicy.owner import cleanup_owned, same, save

    d = Path(directory)
    if same(read(d / "identity.json")) or (d / "released.json").exists():
        return False
    targets = cleanup_owned(d)
    receipt = {
        "time": time.time(),
        "reason": "owner exited without release",
        "cleaned": targets,
    }
    save(d / "abrupt-exit-cleanup.json", receipt)
    save(d / "released.json", receipt)
    return True


def watch_owner(directory):
    from tools.dojo_offpolicy.owner import exact_signal, gpus, same, save

    d = Path(directory)
    identity = read(d / "identity.json")
    gpu = read(d / "plan.json")["gpu"]
    empty_since = None
    while same(identity):
        now = time.time()
        status = read(d / "status.json")
        phase = status.get("phase", "")
        if phase not in ("smoke", "smoke_resume", "formal"):
            empty_since = None  # Native startup already has a 600-second bound.
        else:
            try:
                empty = not gpus()[gpu]["processes"]
            except Exception as error:
                save(
                    d / "fallback-watch-error.json", {"time": now, "error": str(error)}
                )
                empty = False  # An unavailable GPU query is not an empty card.
            empty_since = (empty_since or now) if empty else None
            progress = max(
                [status.get("started", status.get("time", now))]
                + [
                    p.stat().st_mtime
                    for p in (d / f"{phase}.log", d / f"native-g{gpu}/progress.json")
                    if p.exists()
                ]
            )
            if abnormal_empty(now, status.get("started", now), progress, empty_since):
                save(
                    d / "abnormal-idle.json",
                    {
                        "time": now,
                        "phase": phase,
                        "empty_since": empty_since,
                        "last_progress": progress,
                    },
                )
                exact_signal(identity, signal.SIGTERM)
                deadline = time.monotonic() + 45
                while same(identity) and time.monotonic() < deadline:
                    time.sleep(1)
                if same(identity):
                    exact_signal(identity, signal.SIGKILL)
                while same(identity):
                    time.sleep(0.25)
                break
        time.sleep(5)
    release_missing(d)


def latest_checkpoint(plan):
    root = Path(plan["run"]) / "formal/formal/checkpoints"
    candidates = [
        p
        for p in root.glob("global_step_*")
        if (p / "dojo-complete.json").is_file()
        and (p / "actor/local_shard_checkpoint/checkpoint_rank_0.pt").is_file()
        and (p / "actor/sac_components/dsrl_trainer_state_rank_0.pt").is_file()
    ]
    return max(candidates, key=lambda p: int(p.name.rsplit("_", 1)[1]), default=None)


def prepare_retry(old, accepted, attempt):
    new = accepted.with_name(accepted.name + f"-recovery-{attempt}")
    new.mkdir(mode=0o700)
    (new / "config").mkdir()
    (new / "rpc").mkdir(mode=0o700)
    gpu = read(old / "plan.json")["gpu"]
    for name in (
        "plan.json",
        "training-environment.json",
        "sim-environment.json",
        f"native-g{gpu}.json",
        "config/formal.yaml",
    ):
        (new / name).write_text((old / name).read_text().replace(str(old), str(new)))
    (new / "rpc/authkey").write_bytes((old / "rpc/authkey").read_bytes())
    (new / "rpc/authkey").chmod(0o600)
    plan = read(new / "plan.json")
    # A one-time prepared swap is complete; retry through the normal RLT handoff.
    plan.pop("pre_registered", None)
    plan.pop("predecessors", None)
    plan["phases"] = [p for p in plan["phases"] if p["name"] == "formal"]
    (new / "plan.json").write_text(json.dumps(plan, indent=2))
    invalid = {"train": set(), "eval": set()}
    cfg = read(old / f"native-g{gpu}.json")
    for mode in invalid:
        invalid[mode].update(cfg.get("invalid_layouts", {}).get(mode, []))
    log = old / f"native-g{gpu}/invalid-resets.jsonl"
    for row in map(json.loads, log.read_text().splitlines() if log.exists() else []):
        invalid[row["mode"]].update(row["invalid_layouts"])
    cfg = read(new / f"native-g{gpu}.json")
    cfg["invalid_layouts"] = {k: sorted(v) for k, v in invalid.items()}
    (new / f"native-g{gpu}.json").write_text(json.dumps(cfg, indent=2))
    path = new / "config/formal.yaml"
    cfg = yaml.safe_load(path.read_text())
    checkpoint = latest_checkpoint(plan)
    cfg["runner"]["resume_dir"] = str(checkpoint) if checkpoint else None
    cfg["env"]["train"]["seed"] += 104729
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    (new / "recovery-receipt.json").write_text(
        json.dumps(
            {
                "previous": str(old),
                "attempt": attempt,
                "resume": str(checkpoint) if checkpoint else None,
                "discarded": "unfinished wave only",
                "time": time.time(),
            },
            indent=2,
        )
    )
    return new, plan


def main(deployment):
    accepted = Path(deployment).resolve()
    lock = (accepted / "recovery.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ops = Path("/data/chenyiteng/deployment-20261008/bc-signal-tau-v1")
    sys.path.insert(0, str(ops / "ops"))
    from common import proc, save

    save(accepted / "recovery-identity.json", proc(os.getpid()))
    current = accepted
    for attempt in range(4):
        watch_owner(current)
        if read(current / "status.json").get("phase") == "COMPLETE":
            return
        if (
            attempt == 3
            or not (current / "released.json").is_file()
            or not (accepted / "smoke_resume-complete.json").is_file()
            or not simulation_failure(current, ("formal.log",))
        ):
            return
        new, plan = prepare_retry(current, accepted, attempt + 1)
        deadline = time.monotonic() + 150
        while read(ops / "status.json")["slots"][str(plan["gpu"])]["phase"] not in (
            "RLT_RUNNING",
            "RLT_EXITED",
            "WAITING_RLT",
        ):
            assert time.monotonic() < deadline, "RLT handback not ready"
            time.sleep(1)
        env = dict(
            os.environ,
            **read(new / "training-environment.json"),
            CUDA_VISIBLE_DEVICES="",
        )
        with (new / "outer.log").open("x") as log:
            child = subprocess.Popen(
                [
                    plan["python"],
                    "-u",
                    "-B",
                    str(
                        Path(plan["training_source"]) / "tools/dojo_offpolicy/owner.py"
                    ),
                    str(new),
                ],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        while not (new / "identity.json").exists():
            assert child.poll() is None
            time.sleep(0.2)
        save(
            accepted / "recovery-current.json",
            {"deployment": str(new), "attempt": attempt + 1, "time": time.time()},
        )
        current = new


if __name__ == "__main__":
    main(sys.argv[1])
