# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Exact single-slot preflight followed by the established train owner/fallback."""

import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, "/data/chenyiteng/deployment-20261008/bc-signal-tau-v1/ops")
from common import admission_card, exact_signal, gpus, proc, read, same, save


def main(directory):
    d = Path(directory)
    p = read(d / "plan.json")
    root = Path(p["training_source"])
    actual = Path(p.get("deployment", str(d)))
    me = proc(os.getpid())
    gpu = int(p["gpu"])
    assert os.getuid() == 1003 and gpu in (4, 5, 6, 7)
    save(d / "identity.json", me)
    save(d / "status.json", {"phase": "WAIT_PREPARED_SWAP", "time": time.time()})
    child = None

    def stop(*_):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, stop)
    try:
        if p.get("from_rlt"):
            sys.path.insert(0, str(root / "tools/rlt_disk_fallback"))
            from priority_pair_handoff import handoff

            h = d / (actual.name + "-handoff-" + str(os.getpid()))
            h.mkdir()
            handoff(
                h,
                [gpu],
                me,
                reservations={
                    str(gpu): {"identity": me, "release": str(d / "released.json")}
                },
            )
            save(d / "registered.json", {"identity": me, "time": time.time()})
        while not (d / "registered.json").exists():
            time.sleep(0.2)
        assert read(d / "registered.json")["identity"] == me
        deadline = time.monotonic() + 600
        while (
            any(same(x) for x in p.get("predecessors", []))
            or admission_card(gpus()[gpu], p.get("gpu_admission", {}))["processes"]
        ):
            assert time.monotonic() < deadline, "Previous task cleanup deadline"
            time.sleep(0.5)
        env = dict(os.environ, **read(actual / "training-environment.json"))
        env.update(
            CUDA_VISIBLE_DEVICES=str(gpu),
            PYTHONPATH=str(root),
            LOCAL_RANK="0",
            DOJO_GRPO_OWNER=str(d),
            OMP_NUM_THREADS="2",
        )
        commands = [
            (
                "fp32_model",
                [
                    p["python"],
                    "-u",
                    "-B",
                    str(root / "tools/dsrl_fp32_accel/check_model.py"),
                    str(actual),
                ],
                900,
            )
        ]
        if p.get("acceleration_trial"):
            commands.append(
                (
                    "accel_components",
                    [
                        p["python"],
                        "-u",
                        "-B",
                        str(root / "tools/dsrl_fp32_accel/check_small.py"),
                        "--root",
                        str(root),
                        "--device",
                        "cuda",
                        "--output",
                        str(actual / "component-timings.json"),
                    ],
                    1200,
                )
            )
        for phase, command, timeout in commands:
            save(d / "status.json", {"phase": phase, "time": time.time()})
            with (actual / (phase + ".log")).open("x") as f:
                child = subprocess.Popen(
                    command,
                    cwd=root,
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            who = proc(child.pid)
            save(d / "preflight-child.json", who)
            assert child.wait(timeout=timeout) == 0, phase + " failed"
            child = None
        if p.get("acceleration_trial"):
            from omegaconf import OmegaConf

            evidence = read(actual / "component-timings.json")
            cases = evidence["benchmarks"][p["bench"]]["q_update_timings"]
            baseline = cases["vec0_fused0_compile0"]["median_seconds"]
            eligible = {
                k: v
                for k, v in cases.items()
                if "error" not in v
                and v.get("max_param_delta_after_105_updates", 0) < 1e-3
            }
            selected = min(eligible, key=lambda k: eligible[k]["median_seconds"])
            if baseline / eligible[selected]["median_seconds"] < 1.05:
                selected = "vec0_fused0_compile0"
            # Compile is first measured here; whole-training acceptance still
            # requires smoke + full-state resume before a fresh formal run.
            vec, fused, compiled = [bool(int(x[-1])) for x in selected.split("_")]
            for path in (actual / "config").glob("*.yaml"):
                c = OmegaConf.load(path)
                c.actor.model.openpi.dsrl_vectorized_q = vec
                c.actor.model.openpi.dsrl_compile_q = compiled
                c.actor.optim.fused = fused
                c.actor.critic_optim.fused = fused
                c.algorithm.entropy_tuning.optim.fused = fused
                c.actor.defer_metrics = True
                OmegaConf.save(c, path)
            save(
                actual / "selected-acceleration.json",
                {
                    "selected": selected,
                    "q_microbench_speedup": baseline
                    / eligible[selected]["median_seconds"],
                    "defer_metrics": True,
                    "scope": "small Q update; end-to-end speed not inferred",
                    "time": time.time(),
                },
            )
        # Same reservation/identity; established owner handles simulators, Ray
        # cleanup, smoke/resume, formal training, and final RLT release.
        sys.path.insert(0, str(root))
        p = read(d / "plan.json")
        p["predecessors"] = []
        p["from_rlt"] = False
        save(d / "plan.json", p)
        save(
            d / "registered.json", {"identity": proc(os.getpid()), "time": time.time()}
        )
        if p["bench"] == "dojo":
            from tools.dojo_offpolicy.owner import main as owner
        else:
            from tools.robotwin_dsrl_full.queue import main as owner
        owner(str(d))
    except BaseException:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if child is not None and child.poll() is None:
            exact_signal(who, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                exact_signal(who, signal.SIGKILL)
        if not (d / "released.json").exists():
            receipt = {
                "phase": "FAILED_RETURN_RLT",
                "time": time.time(),
                "error": traceback.format_exc(),
            }
            save(d / "status.json", receipt)
            save(d / "released.json", receipt)
        raise


if __name__ == "__main__":
    main(sys.argv[1])
