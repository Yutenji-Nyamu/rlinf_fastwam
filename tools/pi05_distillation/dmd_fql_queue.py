# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""One pair reservation: DMD acceptance, real FQL data, FQL train/eval, then RLT.

Reuse the accepted training launcher, resident evaluator and exact-token cleanup.
Never substitute smoke data for the formal offline dataset.
"""

import json
import os
import signal
import subprocess
import time
import traceback
from pathlib import Path

import run_queue as q
from multitask_eval import sweep

D, O = q.D, q.O
owns_pair = False
me = q.proc(os.getpid())


def main():
    global owns_pair
    plan = q.read(D / "schedule.json")
    q.save(D / "identity.json", me)
    q.event("WAIT_HANDOFF", predecessor=plan["predecessor"])
    while True:
        if (D / "registered.json").exists():
            registered = q.read(D / "registered.json")["identity"]
            if all(registered[k] == me[k] for k in ("pid", "uid", "start")):
                break
        time.sleep(1)
    while q.same(plan["predecessor_identity"]) or any(
        q.gpus()[g]["processes"] for g in q.CARDS
    ):
        time.sleep(2)
    owns_pair = True
    q.save(D / "handoff-complete.json", {"time": time.time(), "identity": me})
    smoke = O / "smoke-onedp"
    args = ["--decode-workers", "8", "--prefetch-batches", "1", "--profile-stages"]
    if not (D / "smoke-onedp-accepted.json").exists():
        first = q.train("onedp", smoke, 12, label="dmd-new-12", extra_args=args)
        q.train("onedp", smoke, 14, first["path"], label="dmd-new-resume-14", extra_args=args)
    q.accept_smoke("onedp")
    q.event("DMD_SMOKE_PASSED", next_stage="FQL_REAL_TRANSITION_COLLECTION")

    # 50 tasks x 12 independent batches x 8 episodes = 4,800 planned episodes.
    # Failed simulator batches remain explicit missing data; ordinary failures stay.
    collection = D / "collection"
    configs = q.read(collection / "collection-plan.json")["configs"]
    assert len(configs) == 600
    q.event("FQL_COLLECTING", requested_episodes=4800, tasks=50, total_parallel=48)
    result = sweep(D, "fql-collection", steps=10, collection_configs=configs)
    q.event("FQL_DATA_INDEX", collection=result["episodes"])
    index = subprocess.run(
        [q.PYTHON, "-B", str(q.R / "tools/pi05_distillation/fql_index.py"),
         str(collection / "batches"), "--output", str(collection / "transition-manifest.json"),
         "--config", str(D / "fql_fixed_teacher.json")],
        cwd=q.R, capture_output=True, text=True, check=True,
    )
    (D / "collection-index.log").write_text(index.stdout + index.stderr)
    counts = json.loads(index.stdout.strip().splitlines()[-1])
    assert counts["episodes"] > 0 and counts["transitions"] > 0
    q.save(D / "dataset-ready.json", counts)
    config = q.read(D / "fql_fixed_teacher.json")
    q.save(D / "formal-ready.json", {
        "time": time.time(), "eval_steps": config["checkpoint_keep_steps"],
        "data": counts, "global_batch": config["batch_size"],
        "microbatch_per_rank": config["microbatch_per_rank"], "epochs": config["epochs"],
    })
    resume = None
    for stop in config["checkpoint_keep_steps"]:
        receipt = q.train("fql_fixed_teacher", O / "fql", stop, resume,
                          extra_args=["--check-export"] if resume is None else ())
        q.event("EVALUATING", method="fql_fixed_teacher", step=stop)
        sweep(D, f"fql-s{stop:06d}", policy=receipt["policy"], method="fql_fixed_teacher")
        resume = receipt["path"]
    q.event("COMPLETE", method="dmd_smoke_then_fql")


if __name__ == "__main__":
    code = 0
    try:
        main()
    except BaseException:
        code = 1
        q.event("FAILED", error=traceback.format_exc())
        for identity in q.owned_children:
            if q.same(identity):
                q.exact_signal(identity, signal.SIGTERM)
        raise
    finally:
        deadline = time.monotonic() + 90
        while owns_pair and any(q.gpus()[g]["processes"] for g in q.CARDS) and time.monotonic() < deadline:
            time.sleep(2)
        if owns_pair and all(not q.gpus()[g]["processes"] for g in q.CARDS):
            q.save(D / "released.json", {"time": time.time(), "identity": me,
                                        "exit_code": code, "cards_clear": True})
