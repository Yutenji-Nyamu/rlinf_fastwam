# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Paired original-policy C/N sweep, using accepted task lanes and RLT handoff."""

import os
import signal
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, "/data/chenyiteng/deployment-20261008/bc-signal-tau-v1/ops")

from common import gpus, proc, read, save
from multitask_eval import sweep


def main(deployment):
    deployment = Path(deployment)
    prepared = read(deployment / "prepared.json")
    cards = prepared["physical_gpus"]
    me = proc(os.getpid())
    save(deployment / "identity.json", me)

    def event(phase, **fields):
        save(deployment / "status.json", dict(time=time.time(), phase=phase, **fields))

    def stop(*_):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not (deployment / "registered.json").is_file():
        time.sleep(1)
    assert read(deployment / "registered.json")["identity"] == me
    exit_code = 0
    try:
        assert all(not gpus()[g]["processes"] for g in cards)
        # Smoke checks both cards and both N values on real full-length tasks.
        # Formal execution then reuses the accepted three lanes per card.
        tasks = read(deployment / "task-list.json")["tasks"]
        smoke_tasks = [t for t in tasks if t["task"] in ("adjust_bottle", "click_bell")]
        assert len(smoke_tasks) == 2
        for c, n in [(10, 1), (40, 10)]:
            label = f"smoke-c{c}-n{n}"
            event("SMOKE", condition=label)
            s = sweep(
                deployment,
                label,
                tasks=smoke_tasks,
                steps=n,
                method="endpoint",
                overrides={
                    "chunk_sweep": True,
                    "execute_chunk": c,
                    "seed_batch": 0,
                    "tasks_per_gpu": 1,
                },
            )
            assert s["valid_tasks"] == 2 and s["episodes"] == 16, s
        save(deployment / "smoke-accepted.json", {"time": time.time(), "passed": True})
        # Interleave paired N1/N10 and cover every C before moving to new seeds.
        for batch in range(4):
            for c in (10, 20, 30, 40):
                for n in (1, 10):
                    label = f"raw-c{c}-n{n}-b{batch}"
                    event(
                        "EVALUATING",
                        condition=label,
                        completed_sweeps=batch * 8 + (c // 10 - 1) * 2 + (n == 10),
                        total_sweeps=32,
                    )
                    s = sweep(
                        deployment,
                        label,
                        steps=n,
                        method="endpoint",
                        overrides={
                            "chunk_sweep": True,
                            "execute_chunk": c,
                            "seed_batch": batch,
                        },
                    )
                    event(
                        "POINT_BATCH_COMPLETE",
                        condition=label,
                        successes=s["successes"],
                        episodes=s["episodes"],
                        elapsed_seconds=s["elapsed_seconds"],
                    )
        event("COMPLETE", total_sweeps=32, planned_episodes=12800)
    except BaseException:
        exit_code = 1
        event("FAILED", error=traceback.format_exc())
        raise
    finally:
        if all(not gpus()[g]["processes"] for g in cards):
            save(
                deployment / "released.json",
                {
                    "time": time.time(),
                    "identity": me,
                    "exit_code": exit_code,
                    "cards_clear": True,
                },
            )


if __name__ == "__main__":
    main(sys.argv[1])
