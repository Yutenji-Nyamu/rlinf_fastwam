# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Reuse accepted RLT handoff / external-owner replacement on exactly GPUs 4-7."""

import os
import signal
import sys
import time
import traceback
from pathlib import Path

from prepare import ACTIVE, BASE, EXPORT, SCHEDULER, read, save

sys.path.insert(0, str(SCHEDULER / "ops"))
from common import exact_signal, same

sys.path.insert(0, str(EXPORT / "dsrl"))
from tools.dojo_offpolicy import swap_4567 as swap

PY = "/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python"
PAIR = Path("/data/chenyiteng/deployment-20261104/parallel-u47")


def status(phase, **values):
    save(BASE / "launch-status.json", {"phase": phase, "time": time.time(), **values})


def main():
    assert os.getuid() == 1003
    assert not (BASE / "launch-receipt.json").exists()
    for method in ["rlt", "dsrl"]:
        assert read(BASE / f"runtime-check-{method}.json")["result"] == "pass"
    entries = read(BASE / "prepared-dsrl.json")["entries"]
    state, scheduler = read(SCHEDULER / "status.json"), read(SCHEDULER / "plan.json")
    assert state["slots"]["4"]["phase"] == "RLT_RUNNING"
    assert state["slots"]["4"]["identity"] == entries["4"]["predecessor"]
    for g, e in entries.items():
        assert same(e["predecessor"]), ("predecessor changed", g)
        assert not (Path(e["deployment"]) / "identity.json").exists()
        if g != "4":
            assert scheduler["slots"][g]["external_identity"] == e["predecessor"]
    # Retarget the CPU-only waiting join first. It must not claim 4/7 while
    # this newly authorized retry is being installed.
    join = read(PAIR / "join-identity.json")
    assert same(join) and read(PAIR / "join-status.json")["phase"] == "WAIT_BOTH_U"
    jp = read(PAIR / "join-plan.json")
    assert jp["jobs"] == [entries["4"]["old_owner"], entries["7"]["old_owner"]]
    save(BASE / "join-before.json", jp)
    exact_signal(join, signal.SIGTERM)
    deadline = time.monotonic() + 15
    while same(join):
        assert time.monotonic() < deadline
        time.sleep(0.1)
    jp["jobs"] = [entries["4"]["owner"], entries["7"]["owner"]]
    save(PAIR / "join-plan.json", jp)
    swap.R = EXPORT / "dsrl"
    next_join = swap.launch(
        PAIR,
        [
            PY,
            "-u",
            "-B",
            str(swap.R / "tools/robotwin_dsrl_full/join_fql.py"),
            str(PAIR),
        ],
        name="join-retry-20261108.log",
        identity=None,
        cwd=swap.R,
    )
    fp = Path(jp["fql"]) / "formal-plan.json"
    fplan = read(fp)
    fplan["predecessor_identity"] = next_join
    save(fp, fplan)
    status("START_GPU4_FROM_RLT", join=next_join)
    e = entries["4"]
    root, d = Path(e["source"]), Path(e["owner"])
    swap.R = root
    owner4 = swap.launch(
        d,
        [PY, "-u", "-B", str(root / "tools/dsrl_rlt_warmstart/owner.py"), str(d)],
        cwd=root,
    )
    active = read(ACTIVE)
    active["slots"]["4"].update(
        {k: e[k] for k in ["deployment", "owner", "run", "source"]}, identity=owner4
    )
    active["updated"] = time.time()
    save(ACTIVE, active)
    swap.launch(
        d,
        [
            PY,
            "-u",
            "-B",
            "-c",
            f"from tools.dojo_offpolicy.recover_queue import watch_owner; watch_owner({str(d)!r})",
        ],
        name="fallback-watch.log",
        identity=None,
        cwd=root,
    )
    deadline = time.monotonic() + 240
    while True:
        assert same(owner4), "GPU4 returned to fallback; inspect before other cuts"
        if read(SCHEDULER / "plan.json")["slots"]["4"][
            "external_identity"
        ] == owner4 and same(read(SCHEDULER / "owner-identity.json")):
            break
        assert time.monotonic() < deadline, "GPU4 handoff did not finish"
        time.sleep(0.5)
    status("GPU4_REGISTERED_REPLACE_567", owner4=owner4)
    from tools.dsrl_rlt_warmstart import cutover

    cutover.B = BASE
    cutover.R = EXPORT / "dsrl"
    cutover.main(["5", "6", "7"])
    active = read(ACTIVE)
    receipt = {
        "time": time.time(),
        "owners": {g: active["slots"][g]["identity"] for g in entries},
        "routes": {g: active["slots"][g]["deployment"] for g in entries},
        "join_jobs": read(PAIR / "join-plan.json")["jobs"],
    }
    save(BASE / "launch-receipt.json", receipt)
    status("ALL_REGISTERED", **receipt)
    print(receipt, flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        status("FAILED_INSPECT_EXISTING_FALLBACK", error=traceback.format_exc())
        raise
