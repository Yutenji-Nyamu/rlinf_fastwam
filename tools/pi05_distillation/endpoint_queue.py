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

from multitask_eval import completed_checkpoint, sweep

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
    handoff = schedule.get("handoff_after_evaluation")
    if handoff:
        registered = q.read(D / "registered.json")
        assert registered["gpus"] == list(q.CARDS) and registered["identity"] == me
        summary_path = O / ("summary-" + handoff["condition"] + ".json")
        q.event(
            "WAIT_EVALUATION_BOUNDARY", predecessor=handoff, physical_gpus=list(q.CARDS)
        )
        while not summary_path.is_file():
            assert q.same(handoff["identity"]), (
                "Predecessor exited before complete evaluation"
            )
            time.sleep(5)
        summary = q.read(summary_path)
        assert summary["attempted_tasks"] == 50 and len(summary["tasks"]) == 50
        if q.same(handoff["identity"]):
            q.exact_signal(handoff["identity"], signal.SIGTERM)
        deadline = time.time() + 180
        while q.same(handoff["identity"]) or any(
            q.gpus()[g]["processes"] for g in q.CARDS
        ):
            assert time.time() < deadline, "Predecessor cleanup incomplete"
            time.sleep(2)
        # Only now change the shared output's environment descriptors: the old
        # evaluator has finished every task and its source remains available.
        for g in q.CARDS:
            env_path = O / f"environment-g{g}.json"
            env = q.read(env_path)
            q.save(D / f"previous-environment-g{g}.json", env)
            env.update(
                REPO_PATH=str(ROOT),
                RLINF_CODE_WORKING_DIR=str(ROOT),
                EMBODIED_PATH=str(ROOT / "examples/embodiment"),
                PYTHONPATH=str(D / "bootstrap")
                + ":"
                + str(ROOT)
                + ":"
                + env["ROBOTWIN_PATH"],
            )
            q.save(env_path, env)
        q.save(
            D / "handoff-complete.json",
            {"time": time.time(), "predecessor": handoff, "summary": str(summary_path)},
        )
    elif schedule.get("predecessor"):
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
    eval_smoke = schedule.get("evaluation_smoke")
    if eval_smoke:
        names = eval_smoke["tasks"]
        tasks = [t for t in q.read(D / "task-list.json")["tasks"] if t["task"] in names]
        assert len(tasks) == len(names) == 2
        q.event("EVALUATION_SMOKE", method=method, tasks=names)
        result = sweep(
            D,
            "smoke-c10",
            policy=eval_smoke["policy"],
            tasks=tasks,
            method=method,
            overrides={"tasks_per_gpu": 1},
        )
        assert result["missing_tasks"] == 0 and result["episodes"] == 16
        q.save(
            D / "evaluation-smoke-accepted.json",
            {"time": time.time(), "result": result},
        )
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
            "episodes_per_task": q.read(D / "eval-settings.json").get(
                "episodes_per_task", 16
            )
            if (D / "eval-settings.json").exists()
            else 16,
            "baseline_source": str(baseline_source),
        },
    )
    resume = None
    completed_step = 0
    if schedule.get("resume_existing"):
        previous = completed_checkpoint(O, method, stops)
        completed_step = previous["step"]
        resume = previous["path"]
        q.event("RESUME_FULL_CHECKPOINT", step=completed_step, checkpoint=resume)
    for stop in stops:
        if stop <= completed_step:
            continue
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
