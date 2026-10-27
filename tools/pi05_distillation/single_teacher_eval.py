# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""One idle card: RLinf pi05 N5/N1, then C10 only for a small observed gap."""

import argparse
import time
import traceback
from pathlib import Path

from multitask_eval import proc, read, save, sweep


def needs_short_chunk(default, one_step, threshold=0.10):
    """Never use incomplete simulator batches to decide the conditional branch."""
    if default["episodes"] != 128 or one_step["episodes"] != 128:
        return False
    return abs(default["mean_success_rate"] - one_step["mean_success_rate"]) < threshold


def main(deployment):
    deployment = Path(deployment)
    prepared = read(deployment / "prepared.json")
    assert len(prepared["physical_gpus"]) == 1
    output = Path(prepared["output"])
    tasks = [
        {"task": f"adjust_bottle-b{i}", "batch": i, "step_limit": 200} for i in range(4)
    ]
    save(deployment / "identity.json", proc(__import__("os").getpid()))
    results = {}

    def evaluate(chunk, steps):
        label = f"rlinf-c{chunk}-n{steps}"
        save(
            deployment / "status.json",
            {"time": time.time(), "phase": "EVALUATING", "condition": label},
        )
        result = sweep(
            deployment,
            label,
            steps=steps,
            tasks=tasks,
            overrides={"single_teacher": True, "execute_chunk": chunk},
        )
        results[label] = result
        return result

    try:
        default = evaluate(50, 5)
        one = evaluate(50, 1)
        short = needs_short_chunk(default, one)
        decision = {
            "time": time.time(),
            "gap_threshold_pp": 10,
            "c10_requested": short,
            "complete_pair": default["episodes"] == one["episodes"] == 128,
            "default_successes": default["successes"],
            "one_step_successes": one["successes"],
        }
        save(output / "conditional-decision.json", decision)
        if short:
            evaluate(10, 5)
            evaluate(10, 1)
        save(
            output / "single-teacher-summary.json",
            {
                "time": time.time(),
                "decision": decision,
                "conditions": {
                    k: {
                        f: v[f]
                        for f in (
                            "episodes",
                            "successes",
                            "mean_success_rate",
                            "missing_tasks",
                            "elapsed_seconds",
                        )
                    }
                    for k, v in results.items()
                },
            },
        )
        save(
            deployment / "status.json",
            {"time": time.time(), "phase": "COMPLETE", "conditions": list(results)},
        )
    except BaseException:
        save(
            deployment / "status.json",
            {"time": time.time(), "phase": "FAILED", "error": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("deployment")
    main(parser.parse_args().deployment)
