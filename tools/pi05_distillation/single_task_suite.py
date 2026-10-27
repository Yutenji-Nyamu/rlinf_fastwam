# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""Bounded five-method smoke reservation, then restore original FQL -> RLT."""

import concurrent.futures
import json
import math
import os
import signal
import statistics
import subprocess
import time
import traceback
from pathlib import Path

import run_queue as q
from multitask_eval import sweep
from single_task_prepare import METHODS, evaluation_config, selected_micro

D, O = q.D, q.O
plan = q.read(D / "single-plan.json")


def pair_evaluation(label, policy=None, method="endpoint", collection=False):
    """Reuse the existing resident evaluator independently on the two cards."""
    jobs = []
    for g in q.CARDS:
        ed = D / label / f"g{g}"
        eo = O / label / f"g{g}"
        eo.mkdir(parents=True, exist_ok=True)
        batch = (32 if g == q.CARDS[0] else 64) if collection else plan["eval_batch"]
        q.save(ed / "prepared.json", {"output": str(eo), "physical_gpus": [g]})
        q.save(eo / f"environment-g{g}.json", q.read(O / f"environment-g{g}.json"))
        q.save(ed / "eval-template.json", q.read(D / "eval-template.json"))
        q.save(ed / "task-list.json", {"tasks": []})
        q.save(
            ed / "eval-settings.json",
            {
                "single_distill": True,
                "episodes_per_task": batch,
                "tasks_per_gpu": 1,
                "resident_model": True,
                "poll_seconds": 0.5,
                "recording_mode": "none",
                "record_decisions": False,
                "execute_chunk": 50,
            },
        )
        paths = []
        count = 64 // batch if collection else 1
        for b in range(count):
            seeds = (
                D
                / "seeds"
                / (
                    f"collect-g{g}-b{b}.json"
                    if collection
                    else f"eval-{batch}-g{g}.json"
                )
            )
            out = eo / "batches" / f"adjust_bottle-b{b}"
            cfg = evaluation_config(
                q.read(D / "eval-template.json"),
                out,
                seeds,
                g,
                batch,
                weights=policy,
                method=method,
                collect=collection,
                forbidden=plan["heldout_seeds"],
            )
            cfg["noise_seed"] += 1000003 * (g * 2 + b)
            p = ed / f"config-b{b}.json"
            q.save(p, cfg)
            paths.append(str(p))
        # A collection-config wrapper also supports supplied inference configs;
        # normal evaluations use the accepted task/seed constructor below.
        if collection:
            kwargs = {"steps": 5, "collection_configs": paths}
        else:
            task = {"task": "adjust_bottle", "batch": g - q.CARDS[0], "step_limit": 200}
            q.save(ed / "eval-seeds/adjust_bottle.json", q.read(seeds))
            kwargs = {"tasks": [task], "policy": policy, "method": method, "steps": 1}
        jobs.append((g, ed, kwargs))
    q.event(
        "SINGLE_COLLECTING" if collection else "SINGLE_EVALUATING",
        label=label,
        method=method,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            (g, pool.submit(sweep, ed, label, **kwargs)) for g, ed, kwargs in jobs
        ]
        results = {str(g): f.result() for g, f in futures}
    q.save(D / f"{label}-results.json", results)
    return results


def collect():
    results = pair_evaluation("native-collection-capacity", collection=True)
    # Compare observed total throughput, not VRAM occupancy alone.
    speed = {
        g: r["episodes"] / r["elapsed_seconds"] if r["episodes"] == 64 else 0
        for g, r in results.items()
    }
    plan["eval_batch"] = 64 if speed[str(q.CARDS[1])] > speed[str(q.CARDS[0])] else 32
    q.save(D / "single-plan.json", plan)
    index = subprocess.run(
        [
            q.PYTHON,
            "-B",
            str(q.R / "tools/pi05_distillation/fql_index.py"),
            str(O / "native-collection-capacity"),
            "--output",
            str(D / "transitions.json"),
        ],
        cwd=q.R,
        env=dict(os.environ, PYTHONPATH=str(q.R)),
        capture_output=True,
        text=True,
        check=True,
    )
    counts = json.loads(index.stdout.strip().splitlines()[-1])
    assert counts["episodes"] >= 32 and counts["transitions"] > 0
    c = q.read(D / "fql_fixed_teacher.json")
    c.update(
        transition_manifest=str(D / "transitions.json"),
        budget_resolved=True,
        epochs=max(
            20, math.ceil(20 / math.ceil(counts["transitions"] / c["batch_size"]))
        ),
        warmup_steps=1,
    )
    q.save(D / "fql_fixed_teacher.json", c)
    q.save(D / "single-transition-receipt.json", counts)


def metrics(output):
    rows = [
        json.loads(s)
        for rank in (0, 1)
        for s in (output / f"metrics-rank{rank}.jsonl").read_text().splitlines()
    ]
    return {
        "peak_gib": max(r["peak_gib"] for r in rows),
        "seconds_per_update": statistics.median(
            r["seconds_per_update"] for r in rows if r["step"] >= 3
        ),
    }


def train(method, output, stop, resume=None, label=None, capacity=False):
    extra = ["--capacity-only"] if capacity else []
    if method != "fql_fixed_teacher":
        extra += [
            "--decode-workers",
            "8",
            "--prefetch-batches",
            "1",
            "--profile-stages",
        ]
    elif not capacity:
        extra += ["--check-export"]
    before = time.time()
    result = q.train(method, output, stop, resume, label, extra_args=extra)
    q.save(D / f"{label}-timing.json", {"started": before, "finished": time.time()})
    return result


def run_method(method):
    accepted = D / f"smoke-{method}-accepted.json"
    evaluated = D / f"export-{method}-results.json"
    if accepted.exists() and evaluated.exists():
        results = q.read(evaluated)
        if (
            q.read(accepted)["passed"]
            and set(results) == {str(g) for g in q.CARDS}
            and all(r["episodes"] == plan["eval_batch"] for r in results.values())
        ):
            q.event("REUSE_COMPLETED_SINGLE_SMOKE", method=method)
            return
    config_path = D / f"{method}.json"
    c = q.read(config_path)
    base = c["microbatch_per_rank"]
    out = O / f"capacity-{method}-{base}"
    train(method, out, 4, label=out.name, capacity=True)
    first = metrics(out)
    chosen, chosen_metrics = base, first
    candidate = selected_micro(method, first)
    tests = [dict(micro=base, **first)]
    if candidate != base:
        c.update(microbatch_per_rank=candidate, batch_size=2 * candidate)
        q.save(config_path, c)
        alt = O / f"capacity-{method}-{candidate}"
        try:
            train(method, alt, 4, label=alt.name, capacity=True)
            second = metrics(alt)
            tests.append(dict(micro=candidate, **second))
            if (
                second["peak_gib"] < 76
                and candidate / second["seconds_per_update"]
                > 0.98 * base / first["seconds_per_update"]
            ):
                chosen, chosen_metrics = candidate, second
        except RuntimeError:
            log = (D / f"{alt.name}.log").read_text(errors="replace")
            if "out of memory" not in log.lower():
                raise
            tests.append({"micro": candidate, "oom": True})
    c.update(microbatch_per_rank=chosen, batch_size=2 * chosen)
    q.save(config_path, c)
    q.save(
        D / f"capacity-{method}.json",
        {"selected": chosen, "trials": tests, "metrics": chosen_metrics},
    )
    output = O / f"smoke-{method}"
    receipt = train(method, output, 12, label=f"{method}-smoke12")
    receipt = train(method, output, 14, receipt["path"], label=f"{method}-resume14")
    if method == "fql_fixed_teacher":
        for rank in (0, 1):
            check = q.read(output / f"export-check-rank{rank}.json")
            assert check["calls"] == 1 and check["finite"] and check["step"] == 14
        q.save(
            D / f"smoke-{method}-accepted.json",
            {"passed": True, "checkpoint": receipt, "metrics": metrics(output)},
        )
    else:
        q.accept_smoke(method)
    result = pair_evaluation(f"export-{method}", receipt["policy"], method)
    assert all(r["episodes"] == plan["eval_batch"] for r in result.values())
    q.event(
        "SINGLE_METHOD_PASSED",
        method=method,
        micro=chosen,
        eval_batch=plan["eval_batch"],
    )


def restore():
    # The next real workload is prepared before pausing anything. This process
    # keeps the RLT reservation until the new FQL owner registers itself.
    old = Path(plan["original_deployment"])
    with (D / "restored-fql.log").open("x") as log:
        child = subprocess.Popen(
            [
                "/usr/bin/python3",
                "-u",
                "-B",
                str(q.R / "tools/pi05_distillation/resume_existing_fql.py"),
                str(old),
                plan["original_source"],
            ],
            cwd=q.R,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    q.save(
        D / "returned-to-fql.json", {"time": time.time(), "identity": q.proc(child.pid)}
    )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError("FQL restoration process exited; see restored-fql.log")
        p = old / "single-task-return.json"
        if p.exists() and q.read(p)["identity"]["pid"] == child.pid:
            q.event("RETURNED_TO_MULTITASK_FQL", identity=q.proc(child.pid))
            return
        time.sleep(0.25)
    raise TimeoutError("FQL restoration did not acknowledge ownership")


def main():
    me = q.proc(os.getpid())
    q.save(D / "identity.json", me)
    q.event("PREPARED_WAIT_HANDOFF")
    while not (D / "registered.json").exists():
        time.sleep(0.25)
    assert q.read(D / "registered.json")["identity"]["pid"] == me["pid"]
    while q.same(plan["predecessor_identity"]) or any(
        q.gpus()[g]["processes"] for g in q.CARDS
    ):
        time.sleep(0.25)
    q.save(D / "handoff-complete.json", {"time": time.time(), "identity": me})
    try:
        if not (D / "single-transition-receipt.json").exists():
            collect()
        else:
            q.event(
                "REUSE_COMPLETE_SINGLE_TRANSITIONS",
                receipt=q.read(D / "single-transition-receipt.json"),
            )
        for method in METHODS:
            run_method(method)
        q.save(
            D / "all-smokes-complete.json", {"time": time.time(), "methods": METHODS}
        )
    except BaseException:
        q.save(
            D / "smoke-failure.json",
            {"time": time.time(), "error": traceback.format_exc()},
        )
        raise
    finally:
        # Healthy stages have already reaped their workers. On unexpected errors,
        # stop only our exact run token before returning the reservation.
        for identity in q.owned_children:
            if q.same(identity):
                q.exact_signal(identity, signal.SIGTERM)
        deadline = time.monotonic() + 60
        while (
            any(q.gpus()[g]["processes"] for g in q.CARDS)
            and time.monotonic() < deadline
        ):
            time.sleep(0.5)
        assert all(not q.gpus()[g]["processes"] for g in q.CARDS), (
            "Owned smoke process has not exited"
        )
        restore()


if __name__ == "__main__":
    main()
