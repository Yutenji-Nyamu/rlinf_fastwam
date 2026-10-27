# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""Resume the original multitask FQL, its exact source/config and partial evals."""

import os
import sys
import time
import traceback
from pathlib import Path

# Args: original deployment, original source. run_queue reads argv[1].
sys.path.insert(0, str(Path(sys.argv[2]) / "tools/pi05_distillation"))
import run_queue as q
from multitask_eval import sweep


def resume_evaluation(receipt):
    label = f"fql-s{receipt['step']:06d}"
    target = q.O / f"summary-{label}.json"
    if target.exists():
        return q.read(target)
    old_progress = q.D / ("eval-" + label) / "progress.json"
    finished = q.read(old_progress)["results"] if old_progress.exists() else []
    done = {r["task"] for r in finished}
    tasks = q.read(q.D / "task-list.json")["tasks"]
    pending = [r for r in tasks if r["task"] not in done]
    started = time.time()
    if pending:
        result = sweep(
            q.D,
            label + f"-continued-{int(started)}",
            policy=receipt["policy"],
            method="fql_fixed_teacher",
            tasks=pending,
        )
        finished += result["tasks"]
    good = [r for r in finished if r["valid"]]
    assert len({r["task"] for r in finished}) == len(tasks)
    result = {
        "time": time.time(),
        "condition": label,
        "attempted_tasks": len(tasks),
        "valid_tasks": len(good),
        "missing_tasks": len(tasks) - len(good),
        "episodes": sum(r["episodes"] for r in good),
        "successes": sum(r["successes"] for r in good),
        "episodes_per_task": 8,
        "tasks": finished,
        "mean_success_rate": sum(r["successes"] / r["episodes"] for r in good)
        / len(good)
        if good
        else None,
        "resumed_from_completed_tasks": len(done),
        "elapsed_resume_seconds": time.time() - started,
    }
    q.save(target, result)
    return result


def main():
    me = q.proc(os.getpid())
    q.save(q.D / "identity.json", me)
    q.save(
        q.D / "registered.json",
        {"time": time.time(), "identity": me, "gpus": list(q.CARDS)},
    )
    release = q.D / "released.json"
    if release.exists():
        release.rename(
            q.D / f"released-before-single-smoke-resume-{int(time.time())}.json"
        )
    rlt = Path("/data/chenyiteng/deployment-20261008/bc-signal-tau-v1/plan.json")
    plan = q.read(rlt)
    for g in q.CARDS:
        plan["slots"][str(g)].update(
            external_identity=me, external_release=str(release)
        )
    q.save(rlt, plan)
    receipt = q.read(q.O / "fql/checkpoint.json")
    c = q.read(q.D / "fql_fixed_teacher.json")
    assert Path(receipt["path"]).is_file() and Path(receipt["policy"]).is_file()
    q.event("RESTORED_AFTER_SINGLE_TASK_SMOKES", step=receipt["step"], identity=me)
    q.save(
        q.D / "single-task-return.json",
        {"time": time.time(), "identity": me, "checkpoint": receipt},
    )
    resume_evaluation(receipt)
    for stop in c["checkpoint_keep_steps"]:
        if stop <= receipt["step"]:
            continue
        receipt = q.train(
            "fql_fixed_teacher",
            q.O / "fql",
            stop,
            receipt["path"],
            label=f"restored-fql-{stop}-{int(time.time())}",
        )
        q.event("EVALUATING", method="fql_fixed_teacher", step=stop)
        resume_evaluation(receipt)
    q.event("COMPLETE", method="fql_fixed_teacher")


if __name__ == "__main__":
    code = 0
    try:
        main()
    except BaseException:
        code = 1
        q.event("RESTORED_FQL_FAILED", error=traceback.format_exc())
        raise
    finally:
        if all(not q.gpus()[g]["processes"] for g in q.CARDS):
            q.save(
                q.D / "released.json",
                {
                    "time": time.time(),
                    "identity": q.proc(os.getpid()),
                    "exit_code": code,
                    "cards_clear": True,
                },
            )
