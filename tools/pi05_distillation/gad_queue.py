# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""GAD validation gate -> existing signal recovery -> two formal methods -> RLT.

Reuse single_formal_queue's training, evaluation, cleanup and failure handling.
The 6/7 pair runs all four smokes; 4/5 waits behind its original healthy job.
"""

import json
import os
import subprocess
import time
from pathlib import Path

import single_formal_queue as base

q, plan = base.q, base.plan
D, O = q.D, q.O


def smoke(method):
    output = Path(plan["smoke_output"]) / ("smoke-gad-" + method)
    output.mkdir(parents=True, exist_ok=True)
    q.save(output / "retention-policy.json", {"mode": "latest_complete_only"})
    config = q.read(D / f"{method}.json")
    config["log_every"] = 1
    q.save(D / f"{method}.json", config)
    extra = ["--decode-workers", "8", "--prefetch-batches", "1", "--profile-stages"]
    try:
        receipt = q.train(
            method, output, 12, label=f"smoke-{method}-12", extra_args=extra
        )
    except RuntimeError:
        log = (D / f"smoke-{method}-12.log").read_text(errors="replace")
        if "out of memory" not in log.lower():
            raise
        base.clear_owned()
        micro = config["microbatch_per_rank"]
        if micro % 2:
            raise ValueError("Cannot halve microbatch without changing global batch")
        config["microbatch_per_rank"] = micro // 2
        q.save(D / f"{method}.json", config)
        receipt = q.train(
            method, output, 12, label=f"smoke-{method}-12-retry", extra_args=extra
        )
    receipt = q.train(
        method,
        output,
        14,
        receipt["path"],
        label=f"smoke-{method}-14",
        extra_args=extra,
    )
    checks = [q.read(output / f"inference-check-rank{r}.json") for r in range(2)]
    if not all(
        c["finite"]
        and c["loaded_export"]
        and c["action_expert_calls"] == 1
        and c["shape"] == [32, 50, 14]
        for c in checks
    ):
        raise RuntimeError("GAD exported inference failed")
    rows = [
        json.loads(l) for l in (output / "metrics-rank0.jsonl").read_text().splitlines()
    ]
    tail = rows[-4:]
    accepted = {
        "method": method,
        "time": time.time(),
        "step": 14,
        "inference": checks,
        "microbatch_per_rank": config["microbatch_per_rank"],
        "batch_size": config["batch_size"],
        "peak_gib": max(r["peak_gib"] for r in rows),
        "stage_seconds": {
            k: sum(r.get("stage_seconds", {}).get(k, 0) for r in tail) / len(tail)
            for k in tail[-1].get("stage_seconds", {})
        },
        "weight_std": tail[-1]["gad_weight_std"],
        "weight_mean": tail[-1]["gad_weight_mean"],
        "resume_verified": True,
    }
    q.save(output / "accepted.json", accepted)
    # Fresh formal runs start from the base; these bounded test weights are temporary.
    removed = []
    for key in ("path", "policy"):
        path = Path(receipt[key])
        if path.parent.resolve() != output.resolve() or path.is_symlink():
            raise ValueError("Unexpected smoke checkpoint path")
        removed.append({"path": str(path), "bytes": path.stat().st_size})
        path.unlink()
    q.save(output / "temporary-weights-removed.json", removed)
    return accepted


def signals():
    deployment = plan.get("signal_recovery")
    if not deployment:
        return
    sd = Path(deployment)
    children = []
    q.event("RESUMING_ORIGINAL_SIGNAL_COLLECTION")
    for gpu in q.CARDS:
        with (sd / f"lane-g{gpu}.log").open("x") as log:
            child = subprocess.Popen(
                [
                    q.PYTHON,
                    "-u",
                    "-B",
                    str(q.R / "tools/pi05_nstep_signal/lane.py"),
                    str(sd),
                    str(gpu),
                ],
                cwd=q.R,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        q.owned_children.append(q.proc(child.pid))
        children.append(child)
    for child in children:
        child.wait()
    q.save(
        D / "signal-recovery-results.json",
        {str(g): q.read(sd / f"lane-g{g}-status.json") for g in q.CARDS},
    )
    base.clear_owned()


def main():
    identity = q.proc(os.getpid())
    q.save(D / "identity.json", identity)
    q.event("PREPARED_WAIT_HANDOFF")
    while not (D / "registered.json").exists():
        time.sleep(0.1)
    assert q.read(D / "registered.json")["identity"]["pid"] == identity["pid"]
    while any(q.same(p) for p in plan.get("predecessor_identities", [])):
        time.sleep(0.25)
    q.wait_for_cards()
    gate = Path(plan["smoke_gate"])
    try:
        if plan.get("run_smokes"):
            outcomes = base.run_stages(
                ["endpoint", "cp_v2", "onedp", "mfd"], smoke, base.failed
            )
            q.save(gate, {"time": time.time(), "methods": outcomes})
            signals()
        while not gate.exists():
            time.sleep(0.5)
        accepted = q.read(gate)["methods"]
        outcomes = {}
        for method in plan["methods"]:
            if accepted[method]["status"] != "complete":
                outcomes[method] = {"status": "not_started_smoke_failed"}
                continue
            config = q.read(D / f"{method}.json")
            config.update(
                log_every=10,
                microbatch_per_rank=accepted[method]["result"]["microbatch_per_rank"],
            )
            q.save(D / f"{method}.json", config)
            destination = O / method
            destination.mkdir(parents=True, exist_ok=True)
            q.save(
                destination / "retention-policy.json", {"mode": "latest_complete_only"}
            )
            result = base.run_stages([method], base.run_method, base.failed)
            outcomes.update(result)
        q.save(D / "queue-complete.json", {"time": time.time(), "outcomes": outcomes})
        q.event("COMPLETE_RLT_NEXT", outcomes=outcomes)
    finally:
        base.clear_owned()
        q.save(
            D / "released.json",
            {"time": time.time(), "identity": identity, "cards_clear": True},
        )


if __name__ == "__main__":
    main()
