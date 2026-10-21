# Copyright 2026 The RLinf Authors and task-reward contributors.
# Licensed under the Apache License, Version 2.0.
"""Run the accepted two-GPU WM driver with terminal TOPReward and SZ2 queue return.

The existing B16 owner's placement, Ray initialization, process catalog and
cleanup are reused. This adapter only supplies SZ2 lifecycle primitives and
the already deployed GPU5 graphics scope; it never restarts shared Ray.
"""

import argparse
import fcntl
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path


def load_runtime(plan):
    sys.path.insert(0, plan["queue_ops"])
    import common

    spec = importlib.util.spec_from_file_location(
        "accepted_b16_owner", plan["existing_owner"]
    )
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)

    def proc(pid):
        ident = common.proc(pid)
        if ident:
            try:
                fields = (
                    (Path("/proc") / str(pid) / "stat")
                    .read_text()
                    .rsplit(")", 1)[1]
                    .split()
                )
                ident["ppid"] = int(fields[1])
            except OSError:
                return None
        return ident

    def gpu_processes(indices):
        return [
            dict(row, gpu=index)
            for index, card in common.gpus().items()
            if index in indices
            for row in card["processes"]
        ]

    old.H = types.SimpleNamespace(
        proc=proc,
        same=common.same,
        atomic=common.save,
        save=common.save,
        now=time.time,
        gpu_processes=gpu_processes,
    )
    old.SCOPE_KEYS = {
        "RLINF_OPENDW_GPU_SCOPE_MANIFEST",
        "PYTHONPATH",
        "LD_PRELOAD",
        "__GL_APPLICATION_PROFILE",
        "HOME",
        "USER",
        "LOGNAME",
        "JAX_PLATFORMS",
        "RLINF_CODE_WORKING_DIR",
    }
    return old, common


def validate_smoke(plan):
    sys.modules.setdefault("tensorflow", None)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    cfg = json.loads(Path(plan["trials"][0]["config"]).read_text())
    log = (
        Path(cfg["runner"]["logger"]["log_path"])
        / cfg["runner"]["logger"]["experiment_name"]
    )
    events = list(log.rglob("events.out.tfevents.*"))
    assert events, "Smoke produced no training metrics"
    metrics = {}
    for directory in {p.parent for p in events}:
        accumulator = EventAccumulator(
            str(directory), size_guidance={"scalars": 1}
        ).Reload()
        for tag in accumulator.Tags()["scalars"]:
            metrics[tag] = accumulator.Scalars(tag)[-1].value
    assert all(math.isfinite(x) for x in metrics.values()), "Nonfinite smoke metrics"
    gradients = [v for k, v in metrics.items() if k.endswith("/grad_norm")]
    assert gradients and max(gradients) > 0, "Smoke did not produce an actor gradient"
    cp = log / "checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"
    assert cp.is_file() and cp.stat().st_size > 1024**3, "Smoke checkpoint missing"
    records = Path(
        plan["services"][0]["argv"][
            plan["services"][0]["argv"].index("--output-dir") + 1
        ]
    )
    rows = [
        json.loads(line)
        for line in (records / "service-events.jsonl").read_text().splitlines()
    ]
    rows = [r for r in rows if r["event"] == "row_completed"]
    latest = [r for r in rows if r["request_id"] == rows[-1]["request_id"]]
    latest.sort(key=lambda r: r["row"])
    assert len(latest) == 16, "Smoke must exercise a real WM B16"
    scores = [r["score_last"] for r in latest]
    assert all(math.isfinite(x) and 0 <= x <= 1 for x in scores)
    ranges = [max(scores[i : i + 8]) - min(scores[i : i + 8]) for i in (0, 8)]
    assert max(ranges) > 0, "Both G8 groups have identical terminal rewards"
    return {
        "metrics": metrics,
        "checkpoint": str(cp),
        "terminal_scores": scores,
        "group_ranges": ranges,
        "service_rows": len(rows),
        "formal_initialization": "Original policy; smoke checkpoint is not reused",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("action", choices=("owner", "driver"))
    parser.add_argument("--key")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    old, common = load_runtime(plan)
    if args.action == "driver":
        # Shared Ray can retain another checkout in its node environment.
        # Keep JAX on CPU, as in the accepted WM runtime, and bind this job's
        # workers to the source tree used by its driver.
        os.environ.update(JAX_PLATFORMS="cpu", RLINF_CODE_WORKING_DIR=plan["repo"])
        old.run_driver(plan, args.key)
        return
    assert os.getuid() == 20001 and os.uname().nodename == "h100-gpu02"
    root = Path(plan["owner_dir"])
    root.mkdir(exist_ok=True)
    lock = (root / "owner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (root / "owner-identity.json").exists(), (
        "Use a new owner output after any previous dispatch"
    )
    old.pidfd_probe()
    common.save(root / "owner-identity.json", old.H.proc(os.getpid()))
    old.add_allowlist(plan)
    service_catalog = old.Catalog(
        root / "service-processes.json", plan["token"] + "_service"
    )
    phase_catalog = None
    terminal = "failed"

    def stop(signum, frame):
        raise RuntimeError("Owner received signal " + str(signum))

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def launch(argv, cwd, env, phase, log, catalog):
        env = dict(env, **{old.TOKEN: catalog.token, old.PHASE: phase})
        with Path(log).open("x") as stream:
            child = subprocess.Popen(
                argv,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        catalog.add(old.H.proc(child.pid), phase, "exact Popen child")
        return child

    try:
        service = plan["services"][0]
        target = root / "services" / service["key"]
        target.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, **service["environment"], CUDA_VISIBLE_DEVICES="5")
        wm = launch(
            service["argv"],
            service["cwd"],
            env,
            "service_wm5",
            target / "service.log",
            service_catalog,
        )
        deadline = time.monotonic() + service["startup_seconds"]
        while True:
            assert wm.poll() is None, "WM service exited while loading"
            try:
                health = old.http(service["url"])
            except (OSError, ValueError):
                health = None
            if health:
                assert (
                    health["ok"] and health["is_offloaded"] and health["pid"] == wm.pid
                )
                assert health["physical_gpu"] == 5 and health["wm_batch_size"] == 16
                common.save(root / "cpu-ready.json", health)
                break
            assert time.monotonic() < deadline, "WM startup deadline"
            time.sleep(3)
        common.save(root / "state.json", {"phase": "WAIT_BORROW", "time": time.time()})
        while not Path(plan["borrow_receipt"]).exists():
            assert wm.poll() is None
            time.sleep(3)
        assert all(not common.gpus()[g]["processes"] for g in (4, 5))
        env = common.read(plan["environment_file"])
        env.update(common.read(plan["graphics_fragment"]))
        for key in old.MASKS:
            env.pop(key, None)
        for trial in plan["trials"]:
            key = trial["key"]
            common.save(root / "state.json", {"phase": key, "time": time.time()})
            phase_catalog = old.Catalog(root / (key + "-processes.json"), plan["token"])
            (root / key).mkdir(exist_ok=True)
            argv = [
                plan["python"],
                "-u",
                "-B",
                __file__,
                "--plan",
                str(args.plan),
                "driver",
                "--key",
                key,
            ]
            child = launch(
                argv, plan["repo"], env, key, root / key / "driver.log", phase_catalog
            )
            phase_plan = dict(plan, trials=[trial])
            rc = old.wait_training(phase_plan, child, wm, phase_catalog)
            common.save(root / key / "result.json", {"exit_code": rc})
            common.save(root / key / "cleanup.json", old.cleanup(plan, phase_catalog))
            phase_catalog = None
            assert rc == 0, "Training exited: " + key
            if key == "smoke":
                common.save(root / "smoke-passed.json", validate_smoke(plan))
        terminal = "completed"
    except BaseException:
        common.save(root / "error.json", {"error": traceback.format_exc()})
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if phase_catalog is not None:
            common.save(root / "driver-cleanup.json", old.cleanup(plan, phase_catalog))
        common.save(root / "service-cleanup.json", old.cleanup(plan, service_catalog))
        common.save(root / "final.json", {"status": terminal, "time": time.time()})
        if Path(plan["borrow_receipt"]).exists():
            assert all(not common.gpus()[g]["processes"] for g in (4, 5))
            common.save(
                plan["release_receipt"],
                {
                    "status": terminal,
                    "gpus": [4, 5],
                    "identity": old.H.proc(os.getpid()),
                    "cards_clear": True,
                    "time": time.time(),
                },
            )


if __name__ == "__main__":
    main()
