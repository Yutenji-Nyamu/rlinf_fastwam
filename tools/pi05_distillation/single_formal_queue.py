# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""Native single-task methods in series, two GPUs each, then existing RLT.

Usage: python single_formal_queue.py <prepared-deployment>
All GPU admission, model loading and rendering reuse the accepted smoke path.
"""

import concurrent.futures
import json
import os
import signal
import subprocess
import time
import traceback
from pathlib import Path

import run_queue as q
from multitask_eval import sweep
from single_formal_plan import batch_indices, formal_config, run_stages
from single_task_prepare import evaluation_config

D, O = q.D, q.O
plan = q.read(D / "formal-plan.json")


def pair_evaluation(label, policy=None, method="endpoint", collection=False):
    """32 environments/card, two waves each, no video or extra signal work."""
    jobs = []
    for index, gpu in enumerate(q.CARDS):
        ed, eo = D / label / f"g{gpu}", O / label / f"g{gpu}"
        eo.mkdir(parents=True, exist_ok=True)
        q.save(ed / "prepared.json", {"output": str(eo), "physical_gpus": [gpu]})
        q.save(eo / f"environment-g{gpu}.json", q.read(O / f"environment-g{gpu}.json"))
        q.save(ed / "eval-template.json", q.read(D / "eval-template.json"))
        q.save(ed / "task-list.json", {"tasks": []})
        q.save(
            ed / "eval-settings.json",
            {
                "single_distill": True,
                "episodes_per_task": 32,
                "tasks_per_gpu": 1,
                "resident_model": True,
                "poll_seconds": 0.25,
                "recording_mode": "none",
                "record_decisions": False,
                "execute_chunk": 50,
                "environment_step_timeout_seconds": 1200,
            },
        )
        tasks, paths = [], []
        for batch in batch_indices(index):
            name = f"adjust_bottle-b{batch}"
            seeds = D / "seeds" / f"{'collect' if collection else 'eval'}-b{batch}.json"
            if collection:
                cfg = evaluation_config(
                    q.read(D / "eval-template.json"),
                    eo / "batches" / name,
                    seeds,
                    gpu,
                    32,
                    collect=True,
                    forbidden=plan["heldout_seeds"],
                )
                cfg.update(batch=batch, noise_seed=cfg["noise_seed"] + 1000003 * batch)
                path = ed / f"config-b{batch}.json"
                q.save(path, cfg)
                paths.append(str(path))
            else:
                q.save(ed / "eval-seeds" / f"{name}.json", q.read(seeds))
                tasks.append({"task": name, "batch": batch, "step_limit": 200})
        kwargs = (
            {"steps": 5, "collection_configs": paths}
            if collection
            else {
                "tasks": tasks,
                "policy": policy,
                "method": method,
                "steps": 1,
            }
        )
        jobs.append((gpu, ed, kwargs))
    q.event("COLLECTING" if collection else "EVALUATING", method=method, label=label)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [(g, pool.submit(sweep, ed, label, **kw)) for g, ed, kw in jobs]
        results = {str(g): f.result() for g, f in futures}
    episodes = sum(r["episodes"] for r in results.values())
    successes = sum(r["successes"] for r in results.values())
    summary = {
        "time": time.time(),
        "method": method,
        "condition": label,
        "requested": 128,
        "episodes": episodes,
        "successes": successes,
        "rate": successes / episodes if episodes else None,
        "complete": episodes == 128,
        "cards": results,
    }
    q.save(O / f"summary-{label}.json", summary)
    return summary


def collect_fql():
    """Collect the formal native N5 replay; never train on smoke-only records."""
    result = pair_evaluation("fql-native-collection", collection=True)
    if result["episodes"] < 96:
        raise RuntimeError("Fewer than three valid 32-episode collection batches")
    manifest = D / "transitions.json"
    completed = subprocess.run(
        [
            q.PYTHON,
            "-B",
            str(q.R / "tools/pi05_distillation/fql_index.py"),
            str(O / "fql-native-collection"),
            "--output",
            str(manifest),
        ],
        cwd=q.R,
        env=dict(os.environ, PYTHONPATH=str(q.R)),
        capture_output=True,
        text=True,
        check=True,
    )
    counts = json.loads(completed.stdout.strip().splitlines()[-1])
    if not 0 < counts["successes"] < counts["episodes"]:
        raise RuntimeError("Formal replay has no observed success/failure mixture")
    c = formal_config(
        q.read(D / "fql_fixed_teacher.json"), transitions=counts["transitions"]
    )
    c["transition_manifest"] = str(manifest)
    q.save(D / "fql_fixed_teacher.json", c)
    q.save(D / "fql-data-receipt.json", counts)


def clear_owned():
    """End only exact identities/tokens owned by this queue, never another job."""
    for identity in q.owned_children:
        if q.same(identity):
            q.exact_signal(identity, signal.SIGTERM)
    # Training descendants inherit this exact run token.
    for p in Path("/proc").glob("[0-9]*/environ"):
        try:
            if p.stat().st_uid != os.getuid():
                continue
            if ("DISTILL_RUN_TOKEN=" + D.name).encode() in p.read_bytes().split(b"\0"):
                identity = q.proc(int(p.parent.name))
                if identity and q.same(identity):
                    q.exact_signal(identity, signal.SIGTERM)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    deadline = time.monotonic() + 20
    while (
        any(q.gpus()[g]["processes"] for g in q.CARDS) and time.monotonic() < deadline
    ):
        time.sleep(0.25)
    # sweep handles its own evaluator tokens even when an exception is raised.
    if any(q.gpus()[g]["processes"] for g in q.CARDS):
        for p in Path("/proc").glob("[0-9]*/environ"):
            try:
                if p.stat().st_uid != os.getuid():
                    continue
                env = p.read_bytes().split(b"\0")
                own = ("DISTILL_RUN_TOKEN=" + D.name).encode() in env
                if own:
                    identity = q.proc(int(p.parent.name))
                    if identity and q.same(identity):
                        q.exact_signal(identity, signal.SIGKILL)
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                pass
    deadline = time.monotonic() + 5
    while (
        any(q.gpus()[g]["processes"] for g in q.CARDS) and time.monotonic() < deadline
    ):
        time.sleep(0.25)
    if any(q.gpus()[g]["processes"] for g in q.CARDS):
        raise RuntimeError(
            "GPU pair still occupied; not killing unidentified processes"
        )


def run_method(method):
    if method == "fql_fixed_teacher":
        collect_fql()
    c = q.read(D / f"{method}.json")
    receipt, rows = None, []
    for stop in c["checkpoint_keep_steps"]:
        extra = (
            []
            if method == "fql_fixed_teacher"
            else [
                "--decode-workers",
                "8",
                "--prefetch-batches",
                "1",
                "--profile-stages",
            ]
        )
        receipt = q.train(
            method,
            O / method,
            stop,
            receipt["path"] if receipt else None,
            label=f"{method}-train-{stop}",
            extra_args=extra,
        )
        summary = pair_evaluation(f"{method}-s{stop:06d}", receipt["policy"], method)
        summary["checkpoint"] = receipt
        q.save(O / f"summary-{method}-s{stop:06d}.json", summary)
        rows.append(summary)
        complete = [r for r in rows if r["complete"]]
        if complete:
            best = max(
                complete, key=lambda r: (r["successes"], -r["checkpoint"]["step"])
            )
            q.save(O / f"best-{method}.json", best)
        if len(rows) >= 2 and not rows[-1]["complete"] and not rows[-2]["complete"]:
            raise RuntimeError("Two incomplete evaluation points; skip this method")
    return {"final": receipt, "complete_evaluations": sum(r["complete"] for r in rows)}


def failed(method, exc):
    q.save(
        D / f"failed-{method}.json",
        {"time": time.time(), "error": repr(exc), "traceback": traceback.format_exc()},
    )
    q.event("METHOD_FAILED_ADVANCE", method=method, error=repr(exc))
    clear_owned()


def main():
    me = q.proc(os.getpid())
    q.save(D / "identity.json", me)
    q.event("PREPARED_WAIT_HANDOFF")
    while not (D / "registered.json").exists():
        time.sleep(0.1)
    assert q.read(D / "registered.json")["identity"]["pid"] == me["pid"]
    while q.same(plan["predecessor_identity"]) or any(
        q.gpus()[g]["processes"] for g in q.CARDS
    ):
        time.sleep(0.25)
    q.save(D / "handoff-complete.json", {"time": time.time(), "identity": me})
    try:
        outcomes = run_stages(plan["methods"], run_method, failed)
        q.save(D / "queue-complete.json", {"time": time.time(), "outcomes": outcomes})
        q.event("COMPLETE_RLT_NEXT", outcomes=outcomes)
    finally:
        clear_owned()
        q.save(
            D / "released.json",
            {"time": time.time(), "identity": me, "cards_clear": True},
        )


if __name__ == "__main__":
    main()
