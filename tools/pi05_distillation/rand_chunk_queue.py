# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""One native-randomized smoke, then four paired original-policy sweeps."""

import os
import signal
import sys
import time
import traceback
from pathlib import Path

from multitask_eval import sweep

sys.path.insert(0, "/data/chenyiteng/deployment-20261008/bc-signal-tau-v1/ops")
from common import gpus, proc, read, save


def main(deployment):
    d = Path(deployment)
    p = read(d / "prepared.json")
    cards = p["physical_gpus"]
    me = proc(os.getpid())
    save(d / "identity.json", me)

    def event(phase, **fields):
        save(d / "status.json", dict(time=time.time(), phase=phase, **fields))

    def stop(*_):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not (d / "registered.json").exists():
        time.sleep(1)
    assert read(d / "registered.json")["identity"] == me
    event("WAIT_CARD", physical_gpus=cards)
    while any(gpus()[g]["processes"] for g in cards):
        time.sleep(15)
    code = 1
    try:
        tasks = [
            t
            for t in read(d / "task-list.json")["tasks"]
            if t["task"] == "adjust_bottle"
        ]
        event("SMOKE", task="adjust_bottle", episodes=32, chunk=50, steps=10)
        smoke_path = Path(p["output"]) / "summary-smoke-rand-c50-n10.json"
        s = (
            read(smoke_path)
            if smoke_path.exists()
            else sweep(
                d,
                "smoke-rand-c50-n10",
                tasks=tasks,
                steps=10,
                overrides={
                    "randomized": True,
                    "episodes_per_task": 32,
                    "tasks_per_gpu": 1,
                    "resident_model": False,
                    "execute_chunk": 50,
                },
            )
        )
        save(d / "smoke-accepted.json", s)
        if s["episodes"] == 0 or s["successes"] == 0:
            event("SMOKE_NO_SUCCESS", summary=s)
            code = 0
            return
        for index, (c, n) in enumerate([(10, 1), (10, 10), (50, 1), (50, 10)]):
            label = f"rand-c{c}-n{n}"
            if (Path(p["output"]) / f"summary-{label}.json").exists():
                continue
            event("EVALUATING", condition=label, completed_sweeps=index, total_sweeps=4)
            s = sweep(
                d,
                label,
                steps=n,
                overrides={"randomized": True, "execute_chunk": c},
                completed_rows=read(d / "reused-results.json").get(label, [])
                if (d / "reused-results.json").exists()
                else None,
            )
            event(
                "POINT_COMPLETE",
                condition=label,
                episodes=s["episodes"],
                successes=s["successes"],
                elapsed_seconds=s["elapsed_seconds"],
            )
        code = 0
        event("COMPLETE", planned_episodes=p["planned_episodes"])
    except BaseException:
        event("FAILED", error=traceback.format_exc())
        raise
    finally:
        if all(not gpus()[g]["processes"] for g in cards):
            save(
                d / "released.json",
                {
                    "time": time.time(),
                    "identity": me,
                    "exit_code": code,
                    "cards_clear": True,
                },
            )


if __name__ == "__main__":
    main(sys.argv[1])
