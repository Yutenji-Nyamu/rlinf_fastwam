# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Bounded process recovery, following the accepted dojo_chunk_eval wrapper.

Adopt an already running pair. Only simulation failures after accepted smoke
are retried; discard the unfinished round and restore the latest full actor.
The existing pair owner performs exact cleanup and RLT handback.
"""

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def latest_checkpoint(plan):
    root = Path(plan["run"]) / "clean-dustbin/checkpoints"
    candidates = [
        p
        for p in root.glob("global_step_*")
        if (p / "dojo-complete.json").is_file()
        and (p / "actor/model_state_dict/full_weights.pt").is_file()
    ]
    return max(candidates, key=lambda p: int(p.name.rsplit("_", 1)[1]), default=None)


def simulation_failure(directory):
    error = read(directory / "status.json").get("error", "")
    if "SystemExit: 143" in error:
        return False  # An explicit operator stop must stay stopped.
    for name in ("formal-driver.log", "native-g6.log", "native-g7.log"):
        path = directory / name
        if path.is_file():
            with path.open("rb") as stream:
                stream.seek(max(0, stream.seek(0, 2) - 65536))
                error += stream.read().decode(errors="replace")
    return any(
        marker in error
        for marker in (
            "Physics-invalid rollout",
            "PhysXFatalError",
            "PhysXBrokenError",
            "invalid reset waves",
            "GPU crash detected",
            "ERROR_DEVICE_LOST",
            "Dojo step timed out",
            "Dojo reset timed out",
            "Dojo reset native restart limit exceeded",
        )
    )


def prepare_retry(old, accepted, attempt):
    new = accepted.with_name(accepted.name + f"-recovery-{attempt}")
    new.mkdir(mode=0o700)
    (new / "config").mkdir()
    (new / "rpc").mkdir(mode=0o700)
    names = [
        "plan.json",
        "training-environment.json",
        "sim-environment.json",
        "native-g6.json",
        "native-g7.json",
        "launch-formal.json",
    ]
    for relative in names + [
        f"config/{p.name}" for p in (old / "config").glob("*.yaml")
    ]:
        (new / relative).write_text(
            (old / relative).read_text().replace(str(old), str(new))
        )
    best = old / "best-evaluation.json"
    if best.is_file():
        (new / best.name).write_bytes(best.read_bytes())
    (new / "rpc/authkey").write_bytes((old / "rpc/authkey").read_bytes())
    (new / "rpc/authkey").chmod(0o600)
    plan = read(new / "plan.json")
    plan.update(accepted_smoke=str(accepted), reuse_native_contract=True)
    (new / "plan.json").write_text(json.dumps(plan, indent=2))
    invalid = {"train": set(), "eval": set()}
    for gpu in (6, 7):
        cfg = read(old / f"native-g{gpu}.json")
        for mode in invalid:
            invalid[mode].update(cfg.get("invalid_layouts", {}).get(mode, []))
        log = old / f"native-g{gpu}/invalid-resets.jsonl"
        for row in map(
            json.loads, log.read_text().splitlines() if log.exists() else []
        ):
            invalid[row["mode"]].update(row["invalid_layouts"])
    for gpu in (6, 7):
        (new / f"native-g{gpu}").mkdir()
        (new / f"native-g{gpu}/cache").symlink_to(
            old / f"native-g{gpu}/cache", target_is_directory=True
        )
        path = new / f"native-g{gpu}.json"
        cfg = read(path)
        cfg["invalid_layouts"] = {k: sorted(v) for k, v in invalid.items()}
        path.write_text(json.dumps(cfg, indent=2))
    path = new / "config/formal.yaml"
    cfg = yaml.safe_load(path.read_text())
    checkpoint = latest_checkpoint(plan)
    cfg["runner"]["resume_dir"] = str(checkpoint) if checkpoint else None
    cfg["runner"]["save_interval"] = 1
    # Fresh training layouts after dropping an invalid round; evaluation stays fixed.
    cfg["env"]["train"]["seed"] += 104729
    cfg["algorithm"]["dvac_gradient_weighting"]["output_dir"] = str(new / "diagnostics")
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    (new / "recovery-receipt.json").write_text(
        json.dumps(
            {
                "previous": str(old),
                "attempt": attempt,
                "resume": str(checkpoint) if checkpoint else None,
                "discarded": "unfinished rollout/update since last complete checkpoint",
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
    from common import proc, same, save

    save(accepted / "recovery-identity.json", proc(os.getpid()))
    current = accepted
    for attempt in range(1, 4):  # Dojo upstream also bounds native restarts to three.
        identity = read(current / "identity.json")
        while same(identity):
            time.sleep(10)
        if read(current / "status.json").get("phase") == "COMPLETE":
            return
        if not (current / "released.json").is_file():
            return  # The existing exact cleanup/handback must have finished.
        if not (
            accepted / "smoke-resume-complete.json"
        ).is_file() or not simulation_failure(current):
            return  # Do not loop on model, optimizer, disk, or programming failures.
        new, plan = prepare_retry(current, accepted, attempt)
        deadline = time.monotonic() + 150
        while not all(
            read(ops / "status.json")["slots"][str(g)]["phase"]
            in ("RLT_RUNNING", "RLT_EXITED", "WAITING_RLT")
            for g in (6, 7)
        ):
            if time.monotonic() > deadline:
                raise TimeoutError("Existing RLT handback not ready")
            time.sleep(1)
        with (new / "queue.log").open("x") as stream:
            child = subprocess.Popen(
                [
                    plan["python"],
                    "-u",
                    "-B",
                    str(
                        Path(plan["training_source"]) / "tools/dojo_grpo/pair_owner.py"
                    ),
                    str(new),
                ],
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        while not (new / "identity.json").exists():
            assert child.poll() is None
            time.sleep(0.2)
        save(
            accepted / "recovery-current.json",
            {"deployment": str(new), "attempt": attempt, "time": time.time()},
        )
        current = new


if __name__ == "__main__":
    main(sys.argv[1])
