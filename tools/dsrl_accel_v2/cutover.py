# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Wait for a new complete checkpoint, register replacement, then stop exact owner."""

import fcntl
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

R = Path("/data/chenyiteng/projects/rlinf-shenzhen/worktrees/dsrl-accel-v2-20261105")
B = Path("/data/chenyiteng/deployment-20261105/dsrl-accel-v2")
OLD = B.with_name("dsrl-fp32-accel")
S = Path("/data/chenyiteng/deployment-20261008/bc-signal-tau-v1")
PY = "/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python"
sys.path.insert(0, str(S / "ops"))
sys.path.insert(0, str(R))
from common import exact_signal, read, same, save  # noqa: E402 - pinned server ops
from omegaconf import OmegaConf  # noqa: E402

from tools.dojo_offpolicy import swap_4567 as swap  # noqa: E402 - pinned runtime

swap.R = R


def completed(entry: Any) -> Any:
    """Find the next fully committed checkpoint for this slot."""
    root = Path(entry["run"]) / "formal/formal/checkpoints"
    good = []
    for cp in root.glob("global_step_*"):
        if not (cp / "dojo-complete.json").is_file():
            continue
        required = [
            "actor/local_shard_checkpoint/checkpoint_rank_0.pt",
            "actor/sac_components/replay_buffer/rank_0",
        ]
        if all((cp / x).exists() for x in required):
            good.append(cp)
    return max(good, key=lambda p: int(p.name.rsplit("_", 1)[1])) if good else None


def replace_one(g: Any, entry: Any, checkpoint: Any) -> None:
    """Register a replacement before retiring the exact previous owner."""
    d = Path(entry["deployment"])
    previous = entry["predecessor"]
    assert (
        same(previous)
        and read(S / "plan.json")["slots"][g]["external_identity"] == previous
    )
    c = OmegaConf.load(d / "config/formal.yaml")
    c.runner.resume_dir = str(checkpoint)
    OmegaConf.save(c, d / "config/formal.yaml")
    save(
        d / "continuation.json",
        {
            "checkpoint": str(checkpoint),
            "step": int(checkpoint.name.rsplit("_", 1)[1]),
            "time": time.time(),
        },
    )
    replacement = swap.launch(
        d,
        [PY, "-u", "-B", str(R / "tools/dsrl_accel_v2/phase_guard.py"), str(d)],
        cwd=R,
    )
    pair = Path("/data/chenyiteng/deployment-20261104/parallel-u47")
    join = None
    if g == "7":
        join = read(pair / "join-identity.json")
        assert same(join) and read(pair / "join-status.json")["phase"] == "WAIT_BOTH_U"
        exact_signal(join, signal.SIGTERM)
        deadline = time.monotonic() + 20
        while same(join):
            assert time.monotonic() < deadline
            time.sleep(0.1)
    serial = (S / "priority-handoff.lock").open("a")
    fcntl.flock(serial, fcntl.LOCK_EX)
    ro = read(S / "owner-identity.json")
    assert same(ro)
    exact_signal(ro, signal.SIGTERM)
    deadline = time.monotonic() + 30
    while same(ro):
        assert time.monotonic() < deadline
        time.sleep(0.1)
    lock = (S / "owner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    receipt = d / "cutover"
    receipt.mkdir()
    try:
        p, state = read(S / "plan.json"), read(S / "status.json")
        save(receipt / "before-plan.json", p)
        save(receipt / "before-state.json", state)
        p["slots"][g].update(
            external_identity=replacement,
            external_release=str(d / "released.json"),
            external_blockers=[previous],
        )
        state["slots"][g] = {"phase": "WAIT_EXTERNAL"}
        if g == "6":
            source = {"deployment": str(d), "gpu": 6, "max_mib": 8}
            p["slots"]["4"].setdefault("admission_graphics_sources", []).append(source)
        save(S / "plan.json", p)
        save(S / "status.json", state)
        save(d / "registered.json", {"identity": replacement, "time": time.time()})
    finally:
        lock.close()
        swap.launch(
            receipt,
            [
                "/usr/bin/python3",
                "-u",
                "-B",
                str(S / "ops/owner.py"),
                "--plan",
                str(S / "plan.json"),
            ],
            cwd=S,
            name="rlt-owner.log",
            identity=None,
        )
        serial.close()
    if join:
        jp = read(pair / "join-plan.json")
        jp["jobs"] = [str(d) if x == entry["old_owner"] else x for x in jp["jobs"]]
        save(pair / "join-plan.json", jp)
        new_join = swap.launch(
            pair,
            [
                PY,
                "-u",
                "-B",
                str(R / "tools/robotwin_dsrl_full/join_fql.py"),
                str(pair),
            ],
            cwd=R,
            name=f"join-accel-v2-{d.name}-{replacement['pid']}.log",
            identity=None,
        )
        f = Path(jp["fql"])
        fp = read(f / "formal-plan.json")
        fp["predecessor_identity"] = new_join
        save(f / "formal-plan.json", fp)
    save(
        receipt / "registered.json",
        {
            "replacement": replacement,
            "predecessor": previous,
            "checkpoint": str(checkpoint),
            "time": time.time(),
        },
    )
    exact_signal(previous, signal.SIGTERM)
    swap.launch(
        receipt,
        [
            PY,
            "-u",
            "-B",
            "-c",
            f"from tools.dojo_offpolicy.recover_queue import watch_owner; watch_owner({str(d)!r})",
        ],
        cwd=R,
        name="watch.log",
        identity=None,
    )
    active = read(OLD / "active-runs.json")
    active["time"] = time.time()
    active["slots"][g] = {
        "deployment": str(d),
        "owner": str(d),
        "identity": replacement,
        "source": str(R),
    }
    save(OLD / "active-runs.json", active)
    save(
        d / "cutover-complete.json",
        {
            "time": time.time(),
            "replacement": replacement,
            "checkpoint": str(checkpoint),
        },
    )
    print("CUTOVER", g, checkpoint, replacement, flush=True)


def main() -> None:
    """Wait for complete checkpoints while current training remains active."""
    assert os.getuid() == 1003
    entries = read(B / "prepared.json")["entries"]
    started = time.time()
    minimum = {}
    for g, e in entries.items():
        cp = completed(e)
        minimum[g] = int(cp.name.rsplit("_", 1)[1]) if cp else -1
    save(B / "cutover-wait.json", {"time": started, "minimum_steps": minimum})
    pending = set(entries)
    while pending:
        assert time.time() - started < 7200, "Checkpoint handoff wait limit"
        for g in sorted(pending):
            e = entries[g]
            assert same(e["predecessor"]), ("Predecessor exited", g)
            cp = completed(e)
            if cp is not None and int(cp.name.rsplit("_", 1)[1]) > minimum[g]:
                replace_one(g, e, cp)
                pending.remove(g)
        if pending:
            time.sleep(2)
    save(B / "both-cutovers-complete.json", {"time": time.time()})


if __name__ == "__main__":
    main()
