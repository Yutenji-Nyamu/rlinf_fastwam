# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Finite success-only Dojo scan, retaining one policy and isolating each simulator."""

import copy
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from common import read, save
from layouts import make_plan
from run_task import check_policy_port_available, ops


def compare_probes(a, b):
    rows = []
    for q in (0, 1):
        pa, pb = Path(a) / f"probe-input-q{q}.npz", Path(b) / f"probe-input-q{q}.npz"
        if not pa.exists() or not pb.exists():
            return {"passed": False, "reason": "incomplete paired probe", "rows": rows}
        with np.load(pa) as x, np.load(pb) as y:
            if any(x[k].shape != y[k].shape for k in x.files):
                return {
                    "passed": False,
                    "reason": "active layouts differ",
                    "rows": rows,
                }
            metrics = {
                k: {
                    "max_abs": float(
                        np.max(np.abs(x[k].astype(float) - y[k].astype(float)))
                    ),
                    "mean_abs": float(
                        np.mean(np.abs(x[k].astype(float) - y[k].astype(float)))
                    ),
                }
                for k in x.files
            }
            rows.append({"query": q, "metrics": metrics})
    passed = all(
        r["metrics"]["state"]["max_abs"] < 1e-5
        and r["metrics"]["actions"]["max_abs"] < 2e-3
        and all(
            v["mean_abs"] < 0.1 for k, v in r["metrics"].items() if k.startswith("cam_")
        )
        for r in rows
    )
    return {
        "passed": passed,
        "rows": rows,
        "reason": "same fixed layouts/noise; native physics and success unchanged",
    }


def main(deployment):
    d = Path(deployment)
    p = read(d / "prepared.json")
    o = ops()
    g = p["gpu"]
    source = Path(__file__).resolve().parent
    template = read(d / "template.json")
    output = Path(p["output"])
    me = o.proc(os.getpid())
    save(d / "identity.json", me)
    server = None
    server_id = None
    child = None

    def event(phase, **fields):
        save(d / "status.json", dict(time=time.time(), phase=phase, **fields))

    def halt(*_):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, halt)
    signal.signal(signal.SIGINT, halt)
    while not (d / "registered.json").exists():
        time.sleep(1)
    assert read(d / "registered.json")["identity"] == me
    event("WAIT_CARD", gpu=g)
    while o.gpus()[g]["processes"]:
        time.sleep(15)

    def stop_server():
        nonlocal server, server_id
        if server_id and o.same(server_id):
            o.exact_signal(server_id, signal.SIGTERM)
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                if o.same(server_id):
                    o.exact_signal(server_id, signal.SIGKILL)
                server.wait(timeout=15)
        server = None
        server_id = None
        assert not o.gpus()[g]["processes"], "Card not clear after owned cleanup"

    def start_server():
        nonlocal server, server_id
        if server is not None and server.poll() is None:
            return
        assert not o.gpus()[g]["processes"]
        check_policy_port_available(template["policy"]["port"])
        e = dict(os.environ, **read(template["policy_environment"]))
        e.update(
            CUDA_VISIBLE_DEVICES=str(g),
            DOJO_GPU_SCOPE=str(g),
            DOJO_U_TOKEN=d.name + "-policy",
            DOJO_EVAL_POINTER=str(d / "current-job.json"),
            DOJO_EVAL_DEPLOYMENT=str(d),
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
            XLA_PYTHON_CLIENT_MEM_FRACTION="0.45",
            OMP_NUM_THREADS="1",
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=":".join(
                [
                    template["bootstrap"],
                    str(source),
                    template["repo"],
                    template["repo"] + "/XPolicyLab",
                    template["repo"] + "/XPolicyLab/policy/Pi_05/openpi/src",
                ]
            ),
        )
        with (d / "policy.log").open("a") as log:
            server = subprocess.Popen(
                [
                    template["policy_python"],
                    "-u",
                    "-B",
                    str(source / "policy_entry.py"),
                    str(d / "template.json"),
                ],
                cwd=template["repo"],
                env=e,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        server_id = o.proc(server.pid)
        save(d / "policy-identity.json", server_id)
        limit = time.monotonic() + 900
        while True:
            assert server.poll() is None, "Policy failed during initialization"
            assert time.monotonic() < limit, "Policy loading timeout"
            try:
                with socket.create_connection(
                    ("127.0.0.1", template["policy"]["port"]), timeout=1
                ):
                    break
            except OSError:
                time.sleep(3)

    def task(name, c, n, label, decision_only=False, probe=0):
        nonlocal child
        dest = output / "tasks" / label
        cfg = copy.deepcopy(template)
        cfg.update(
            task=name,
            execute_chunk=c,
            num_steps=n,
            output=str(dest),
            decision_observations_only=decision_only,
            probe_queries=probe,
            layout_plan=make_plan(p["layouts"], name, 8),
        )
        cfg["policy"]["task_name"] = name
        path = d / "jobs" / (label + ".json")
        save(path, cfg)
        save(d / "current-job.json", {"config": str(path)})
        event("PROBE" if probe else "EVALUATING", task=name, condition=label, gpu=g)
        start_server()
        cfg["resident_policy_identity"] = server_id
        save(path, cfg)
        with (d / (label + "-driver.log")).open("x") as log:
            child = subprocess.Popen(
                [sys.executable, "-u", "-B", str(source / "run_task.py"), str(path)],
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        save(d / "child-identity.json", o.proc(child.pid))
        rc = child.wait()
        child = None
        clearance = read(dest / "clearance.json")
        assert clearance["clear"], "Simulator cleanup incomplete"
        row = read(dest / "result.json")
        row["condition"] = label
        if rc == 79 or row["returncode"] != 0:
            stop_server()
        if not probe:
            results = [
                read(x)
                for x in sorted(output.glob("tasks/*/result.json"))
                if read(x)["state"] != "PROBE"
            ]
            save(
                output / "summary.json",
                {
                    "time": time.time(),
                    "completed_jobs": len(results),
                    "planned_jobs": 216,
                    "episodes": sum(x["episodes"] for x in results),
                    "successes": sum(x["successes"] for x in results),
                    "rows": results,
                },
            )
        return row

    try:
        # Two bounded A/B probes; inconclusive equality leaves the old observation path.
        task("stack_bowls", 50, 10, "probe-obs-original", False, 2)
        task("stack_bowls", 50, 10, "probe-obs-boundary", True, 2)
        acceptance = compare_probes(
            output / "tasks/probe-obs-original", output / "tasks/probe-obs-boundary"
        )
        save(d / "observation-acceptance.json", acceptance)
        decision = acceptance["passed"]
        task("stack_bowls", 10, 1, "probe-c10-n1", decision, 2)
        for smoke_name in ("stack_bowls", "stack_blocks"):
            row = task(smoke_name, 50, 10, f"{smoke_name}-c50-n10", decision)
            if row["episodes"] > 0:
                break
        assert row["episodes"] > 0, (
            "Two bounded native task smokes yielded no complete episodes"
        )
        save(
            d / "smoke-accepted.json",
            {
                "time": time.time(),
                "episodes": row["episodes"],
                "successes": row["successes"],
                "decision_observations_only": decision,
                "policy_identity": server_id,
            },
        )
        tasks = read(d / "task-list.json")
        for name in tasks:
            for c, n in [(10, 1), (10, 10), (50, 1), (50, 10)]:
                label = f"{name}-c{c}-n{n}"
                if (output / "tasks" / label / "result.json").exists():
                    continue
                task(name, c, n, label, decision)
        event("COMPLETE", planned_episodes=1728)
    except BaseException:
        event("FAILED", error=traceback.format_exc())
        raise
    finally:
        if child and child.poll() is None:
            o.exact_signal(o.proc(child.pid), signal.SIGTERM)
            child.wait(timeout=90)
        stop_server()
        save(
            d / "released.json",
            {"time": time.time(), "identity": me, "cards_clear": True},
        )


if __name__ == "__main__":
    main(sys.argv[1])
