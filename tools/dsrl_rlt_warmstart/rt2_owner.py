# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Accepted single-slot ownership/cleanup, with bounded optional experiments."""

import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, "/data/chenyiteng/deployment-20261008/bc-signal-tau-v1/ops")
from common import admission_card, exact_signal, gpus, proc, read, same, save


def main(directory: Any) -> None:
    """Own GPU7 through bounded tests, formal training and RLT fallback."""
    d = Path(directory)
    p = read(d / "plan.json")
    me = proc(os.getpid())
    child = None
    assert os.getuid() == 1003 and p["gpu"] in (4, 7)
    save(d / "identity.json", me)
    save(d / "status.json", {"phase": "WAIT_PREPARED_SWAP", "time": time.time()})

    def stop(*_: Any) -> None:
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, stop)
    result = {}
    try:
        if p.get("from_rlt"):
            sys.path.insert(
                0, str(Path(p["training_source"]) / "tools/rlt_disk_fallback")
            )
            from priority_pair_handoff import handoff

            h = d / (d.parent.name + "-" + d.name + "-handoff")
            h.mkdir()
            handoff(
                h,
                [p["gpu"]],
                me,
                reservations={
                    str(p["gpu"]): {"identity": me, "release": str(d / "released.json")}
                },
            )
        else:
            while not (d / "registered.json").exists():
                time.sleep(0.2)
            assert read(d / "registered.json")["identity"] == me
        p = read(d / "plan.json")
        deadline = time.monotonic() + 600
        while (
            any(same(w) for w in p["predecessors"])
            or admission_card(gpus()[p["gpu"]], p.get("gpu_admission", {}))["processes"]
        ):
            assert time.monotonic() < deadline, "Predecessor cleanup timed out"
            time.sleep(0.5)
        env = dict(os.environ, **read(d / "training-environment.json"))
        env.update(
            DOJO_GRPO_DEPLOYMENT=str(d),
            DOJO_GRPO_OWNER=str(d),
            ROBOTWIN_DSRL_OWNER=str(d),
        )
        for phase in p["phases"]:
            save(d / "status.json", {"phase": phase["name"], "time": time.time()})
            with (d / (phase["name"] + ".log")).open("x") as log:
                child = subprocess.Popen(
                    phase["command"],
                    cwd=p["training_source"],
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            who = proc(child.pid)
            save(d / "child.json", who)
            try:
                code = child.wait(timeout=phase.get("timeout"))
            except subprocess.TimeoutExpired:
                exact_signal(who, signal.SIGTERM)
                try:
                    child.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    exact_signal(who, signal.SIGKILL)
                    child.wait(timeout=15)
                code = 124
            child = None
            assert code == 0, (phase["name"], code)
            save(d / (phase["name"] + "-complete.json"), {"time": time.time()})
        result = {"phase": "COMPLETE_RETURN_RLT", "time": time.time()}
    except BaseException:
        result = {
            "phase": "FAILED_RETURN_RLT",
            "time": time.time(),
            "error": traceback.format_exc(),
        }
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if child is not None and child.poll() is None:
            exact_signal(who, signal.SIGTERM)
            try:
                child.wait(timeout=120)
            except subprocess.TimeoutExpired:
                exact_signal(who, signal.SIGKILL)
        from tools.dojo_offpolicy.owner import cleanup_owned

        cleanup_owned(d)
        save(d / "status.json", result)
        save(d / "released.json", result)
