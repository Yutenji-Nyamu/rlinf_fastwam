# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""One short endpoint smoke, or the approved multitask successor of OneDP."""

import importlib.util
import math
import os
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path

from multitask_eval import sweep

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "accepted_queue", ROOT / "tools/pi05_distillation/run_queue.py"
)
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)
D, O = q.D, q.O
me = q.proc(os.getpid())
q.save(D / "identity.json", me)
owns_pair = False


def copy_baselines(source):
    """Reuse the paired CP raw-policy sweeps as each becomes available."""
    for n in [1, 10]:
        path = source / f"summary-raw-n{n}.json"
        if path.exists():
            shutil.copyfile(path, O / path.name)


def main():
    global owns_pair
    if "--smoke-only" in sys.argv[2:]:
        owns_pair = True
        receipt = q.train("endpoint", O / "smoke-endpoint", 12)
        q.train("endpoint", O / "smoke-endpoint", 14, receipt["path"])
        q.accept_smoke("endpoint")
        checks = [
            q.read(O / "smoke-endpoint" / f"native-parity-rank{r}.json") for r in [0, 1]
        ]
        assert all(c["student_initial_N1_matches"] for c in checks)
        q.save(
            D / "smoke-accepted.json",
            {"time": time.time(), "passed": True, "native": checks},
        )
        q.event("SMOKE_COMPLETE", method="endpoint")
        return

    while not (D / "registered.json").is_file():
        time.sleep(1)
    schedule = q.read(D / "schedule.json")
    upstream = Path(schedule["predecessor"])
    q.event("WAIT_ONEDP", predecessor=str(upstream), physical_gpus=list(q.CARDS))
    while not (upstream / "released.json").is_file() or q.same(
        schedule["predecessor_identity"]
    ):
        time.sleep(15)
    assert all(not q.gpus()[g]["processes"] for g in q.CARDS), (
        "Predecessor cleanup incomplete"
    )
    owns_pair = True
    q.save(D / "predecessor-finished.json", q.read(upstream / "released.json"))
    assert q.read(D / "smoke-accepted.json")["passed"]
    data_source = Path(schedule["data_source"])
    shape = q.read(data_source / "data-ready.json")
    assert shape["tasks"] == 50 and shape["episodes"] == 2500
    shutil.copyfile(data_source / "data-manifest.json", D / "data-manifest.json")
    baseline_source = Path(schedule["baseline_source"])
    copy_baselines(baseline_source)
    config = q.read(D / "endpoint.json")
    # The shared data manifest is independent of this method's measured batch.
    # Keep the approved epoch budget when capacity tuning changes update size.
    per_epoch = math.ceil(shape["windows"] / config["batch_size"])
    total = per_epoch * config["epochs"]
    stops = sorted(
        {
            min(total, max(1, round(x * per_epoch)))
            for x in [0.5, 1, 2, 5, 10, 20, 35, 50]
        }
    )
    q.save(
        D / "formal-ready.json",
        {
            "time": time.time(),
            "total_steps": total,
            "eval_steps": stops,
            "epochs": 50,
            "global_batch": config["batch_size"],
            "microbatch_per_rank": config["microbatch_per_rank"],
            "steps_per_epoch": per_epoch,
            "tasks": 50,
            "episodes_per_task": 16,
            "baseline_source": str(baseline_source),
        },
    )
    resume = None
    for stop in stops:
        receipt = q.train("endpoint", O / "endpoint", stop, resume)
        label = f"endpoint-s{stop:06d}"
        copy_baselines(baseline_source)
        q.event("EVALUATING", condition=label, tasks=50)
        sweep(D, label, policy=receipt["policy"], method="endpoint")
        resume = receipt["path"]
    q.event("COMPLETE", method="multitask_endpoint")


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        exit_code = 1
        q.event("FAILED", error=traceback.format_exc())
        for identity in q.owned_children:
            if q.same(identity):
                q.exact_signal(identity, signal.SIGTERM)
        deadline = time.time() + 90
        while (
            owns_pair
            and any(q.gpus()[g]["processes"] for g in q.CARDS)
            and time.time() < deadline
        ):
            time.sleep(2)
        raise
    finally:
        # Cancelling an idle successor must never release its still-active
        # predecessor's GPUs to RLT during a normal model-loading gap.
        if owns_pair and all(not q.gpus()[g]["processes"] for g in q.CARDS):
            q.save(
                D / "released.json",
                {
                    "time": time.time(),
                    "identity": me,
                    "exit_code": exit_code,
                    "cards_clear": True,
                },
            )
