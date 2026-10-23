# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Accepted endpoint/OneDP training with the shared multitask evaluation queue."""

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
prepared = q.read(D / "prepared.json")
method = prepared.get("method", "endpoint")
assert method in {"endpoint", "onedp"}
me = q.proc(os.getpid())
q.save(D / "identity.json", me)
owns_pair = False


def copy_baselines(source):
    """Reuse the paired CP raw-policy sweeps as each becomes available."""
    for n in [1, 10]:
        path = source / f"summary-raw-n{n}.json"
        if path.exists():
            shutil.copyfile(path, O / path.name)


def smoke():
    """Reuse the accepted short training, full resume and native N1 export check."""
    output = O / f"smoke-{method}"
    receipt = q.train(method, output, 12, label=f"smoke-{method}-12")
    q.train(method, output, 14, receipt["path"], label=f"smoke-{method}-14")
    q.accept_smoke(method)
    if method == "endpoint":
        checks = [q.read(output / f"native-parity-rank{r}.json") for r in [0, 1]]
        assert all(c["student_initial_N1_matches"] for c in checks)
    q.save(D / "smoke-accepted.json", {"time": time.time(), "passed": True})


def main():
    global owns_pair
    if "--smoke-only" in sys.argv[2:]:
        owns_pair = True
        smoke()
        q.event("SMOKE_COMPLETE", method=method)
        return

    while not (D / "registered.json").is_file():
        time.sleep(1)
    schedule = q.read(D / "schedule.json")
    if schedule.get("predecessor"):
        upstream = Path(schedule["predecessor"])
        q.event("WAIT_ONEDP", predecessor=str(upstream), physical_gpus=list(q.CARDS))
        while not (upstream / "released.json").is_file() or q.same(
            schedule["predecessor_identity"]
        ):
            time.sleep(15)
        q.save(D / "predecessor-finished.json", q.read(upstream / "released.json"))
    else:
        assert schedule.get("start_after_rlt_handoff") is True
        registered = q.read(D / "registered.json")
        assert registered["gpus"] == list(q.CARDS) and registered["identity"] == me
    assert all(not q.gpus()[g]["processes"] for g in q.CARDS), (
        "Predecessor cleanup incomplete"
    )
    owns_pair = True
    data_source = Path(schedule["data_source"])
    shape = q.read(data_source / "data-ready.json")
    assert shape["tasks"] == 50 and shape["episodes"] == 2500
    shutil.copyfile(data_source / "data-manifest.json", D / "data-manifest.json")
    baseline_source = Path(schedule["baseline_source"])
    copy_baselines(baseline_source)
    config = q.read(D / f"{method}.json")
    # The shared data manifest is independent of this method's measured batch.
    # Keep the approved epoch budget when capacity tuning changes update size.
    per_epoch = math.ceil(shape["windows"] / config["batch_size"])
    total = per_epoch * config["epochs"]
    stops = sorted(
        {
            min(total, max(1, round(x * per_epoch)))
            for x in prepared.get("eval_budget_epochs", [0.5, 1, 2, 5, 10, 20, 35, 50])
        }
    )
    if config.get("checkpoint_keep_last") is not None:
        config["checkpoint_keep_steps"] = stops
        q.save(D / f"{method}.json", config)
    if schedule.get("run_smoke_before_formal"):
        smoke()
    assert q.read(D / "smoke-accepted.json")["passed"]
    q.save(
        D / "formal-ready.json",
        {
            "time": time.time(),
            "total_steps": total,
            "eval_steps": stops,
            "epochs": config["epochs"],
            "method": method,
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
        receipt = q.train(method, O / method, stop, resume)
        label = f"{method}-s{stop:06d}"
        copy_baselines(baseline_source)
        q.event("EVALUATING", condition=label, tasks=50)
        sweep(D, label, policy=receipt["policy"], method=method)
        resume = receipt["path"]
    q.event("COMPLETE", method=f"multitask_{method}")


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
