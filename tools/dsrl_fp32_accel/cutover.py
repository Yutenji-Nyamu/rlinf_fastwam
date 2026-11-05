# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Register replacement and fallback before signalling exact authorized owners."""

import argparse
import fcntl
import json
import os
import signal
import sys
import time
from pathlib import Path

from prepare import BASE as B
from prepare import ROOT as R
from prepare import S

sys.path.insert(0, str(S / "ops"))
from common import exact_signal, read, same, save

sys.path.insert(0, str(R))
from tools.dojo_offpolicy import swap_4567 as swap

swap.R = R
PY = "/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python"


def main(gpus):
    assert os.getuid() == 1003 and set(gpus) <= set("4567")
    receipt = B / ("cutover-" + "".join(gpus))
    receipt.mkdir(exist_ok=False)
    entries = read(B / "prepared-fp32.json")["entries"]
    original = read(S / "plan.json")
    owners = {}
    targets = []
    for g in gpus:
        e = entries[g]
        d = Path(e["owner"])
        assert original["slots"][g]["external_identity"] == e["predecessors"][0], (
            "reservation changed",
            g,
        )
        assert same(e["predecessors"][0]), ("owner changed", g)
        owners[g] = swap.launch(
            d,
            [PY, "-u", "-B", str(R / "tools/dsrl_fp32_accel/slot_owner.py"), str(d)],
            cwd=R,
        )
        targets += e["predecessors"]
    pair = Path("/data/chenyiteng/deployment-20261104/parallel-u47")
    join = None
    if set(gpus) & {"4", "7"}:
        join = read(pair / "join-identity.json")
        assert same(join) and read(pair / "join-status.json")["phase"] == "WAIT_BOTH_U"
        exact_signal(join, signal.SIGTERM)
        deadline = time.monotonic() + 15
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
    try:
        p, state = read(S / "plan.json"), read(S / "status.json")
        save(receipt / "before-plan.json", p)
        save(receipt / "before-state.json", state)
        for g in gpus:
            e = entries[g]
            p["slots"][g].update(
                external_identity=owners[g],
                external_release=str(Path(e["owner"]) / "released.json"),
                external_blockers=e["predecessors"],
            )
            state["slots"][g] = {"phase": "WAIT_EXTERNAL"}
        for g in ("5", "6"):
            source = {
                "deployment": entries[g]["deployment"],
                "gpu": int(g),
                "max_mib": 8,
            }
            sources = p["slots"]["4"].setdefault("admission_graphics_sources", [])
            if source not in sources:
                sources.append(source)
        p4 = Path(entries["4"]["owner"]) / "plan.json"
        four = read(p4)
        four["gpu_admission"] = p["slots"]["4"]
        save(p4, four)
        save(S / "plan.json", p)
        save(S / "status.json", state)
        for g in gpus:
            save(
                Path(entries[g]["owner"]) / "registered.json",
                {"identity": owners[g], "time": time.time()},
            )
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
        save(receipt / "before-join.json", jp)
        for g in set(gpus) & {"4", "7"}:
            jp["jobs"] = [
                entries[g]["owner"] if x == entries[g]["old_owner"] else x
                for x in jp["jobs"]
            ]
        save(pair / "join-plan.json", jp)
        nj = swap.launch(
            pair,
            [
                PY,
                "-u",
                "-B",
                str(R / "tools/robotwin_dsrl_full/join_fql.py"),
                str(pair),
            ],
            cwd=R,
            name="join-fp32-" + "".join(gpus) + ".log",
            identity=None,
        )
        f = Path(jp["fql"])
        fp = read(f / "formal-plan.json")
        fp["predecessor_identity"] = nj
        fp.setdefault("gpu_admission", {})["4"] = p["slots"]["4"]
        save(f / "formal-plan.json", fp)
    save(
        receipt / "prepared.json",
        {"owners": owners, "targets": targets, "time": time.time()},
    )
    for who in targets:
        assert who["uid"] == 1003
        if same(who):
            exact_signal(who, signal.SIGTERM)
    for g in gpus:
        dd = entries[g]["owner"]
        swap.launch(
            receipt,
            [
                PY,
                "-u",
                "-B",
                "-c",
                f"from tools.dojo_offpolicy.recover_queue import watch_owner; watch_owner({dd!r})",
            ],
            cwd=R,
            name=f"watch-g{g}.log",
            identity=None,
        )
    save(
        receipt / "stops-issued.json",
        {"time": time.time(), "owners": owners, "targets": targets},
    )
    print(json.dumps({"registered": owners, "exact_stops": targets}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("gpus", nargs="+", choices=list("4567"))
    main(p.parse_args().gpus)
