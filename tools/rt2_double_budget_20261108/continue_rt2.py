# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Full-state RT2 continuation, adapted from robotwin_dsrl_full/continue_owner.py."""

import fcntl
import os
import signal
import sys
import time
import traceback
from pathlib import Path

S = Path("/data/chenyiteng/deployment-20261008/bc-signal-tau-v1")
ACTIVE = Path("/data/chenyiteng/deployment-20261105/dsrl-fp32-accel/active-runs.json")
sys.path.insert(0, str(S / "ops"))
from common import proc, read, same, save


def checkpoint_ready(path, method):
    """Check each method's real restore format, not just a leftover save marker."""
    p = Path(path)
    common = [
        "dojo-complete.json",
        "actor/sac_components/target_model/checkpoint_rank_0.pt",
    ]
    if method == "dsrl":
        needed = common + [
            "actor/local_shard_checkpoint/checkpoint_rank_0.pt",
            "actor/sac_components/dsrl_trainer_state_rank_0.pt",
            "actor/sac_components/replay_buffer/rank_0/dsrl_transition_replay.pt",
            "actor/sac_components/alpha/dcp_checkpoint/.metadata",
        ]
        shard = p / "actor/sac_components/alpha/dcp_checkpoint"
    elif method == "rlt":
        needed = common + [
            "actor/dcp_checkpoint/.metadata",
            "actor/model_state_dict/full_weights.pt",
            "actor/sac_components/rlt_trainer_state/checkpoint_rank_0.pt",
            "actor/sac_components/rlt_trainer_state/complete.json",
            "actor/sac_components/replay_buffer/rank_0/metadata.json",
        ]
        shard = p / "actor/dcp_checkpoint"
    else:
        raise ValueError(method)
    return all(
        (p / n).is_file() and (p / n).stat().st_size > 0 for n in needed
    ) and any(shard.glob("*.distcp"))


def main(directory):
    d = Path(directory)
    c, p = read(d / "continuation.json"), read(d / "plan.json")
    assert os.getuid() == 1003 and p["gpu"] in (4, 7) and p["bench"] == "rt2"
    me = proc(os.getpid())
    save(d / "identity.json", me)
    save(
        d / "status.json",
        {"phase": "WAIT_CURRENT_FORMAL", "time": time.time(), "target": c["target"]},
    )

    def stop(*_):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, stop)
    try:
        while not (d / "registered.json").exists():
            time.sleep(0.2)
        while same(c["predecessor"]):
            time.sleep(0.5)
        old = Path(c["previous_deployment"])
        assert (old / "released.json").is_file() and (
            old / "formal-complete.json"
        ).is_file(), "Predecessor did not finish normally"
        assert checkpoint_ready(c["checkpoint"], p["method"]), (
            "Final full checkpoint missing; return to RLT"
        )
        assert (
            read(Path(c["checkpoint"]) / "dojo-complete.json")["step"]
            == c["previous_target"]
        )
        # Both continuations may start independently. Preserve the other slot's route.
        with ACTIVE.with_suffix(".continuation.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            active = read(ACTIVE)
            slot = active["slots"][str(p["gpu"])]
            assert slot["owner"] == str(old) and slot["pending_continuation"][
                "owner"
            ] == str(d)
            slot.update(
                deployment=str(d),
                owner=str(d),
                identity=me,
                source=p["training_source"],
                run=p["run"],
            )
            slot.pop("pending_continuation")
            active["updated"] = time.time()
            save(ACTIVE, active)
        save(
            d / "continuation-ready.json",
            {"time": time.time(), "checkpoint": c["checkpoint"], "target": c["target"]},
        )
        # The accepted owner keeps this same PID/reservation and owns cleanup.
        sys.path.insert(0, p["training_source"])
        from tools.dsrl_rlt_warmstart.rt2_owner import main as owner

        owner(str(d))
    except BaseException:
        if not (d / "released.json").exists():
            result = {
                "phase": "FAILED_RETURN_RLT",
                "time": time.time(),
                "error": traceback.format_exc(),
            }
            save(d / "status.json", result)
            save(d / "released.json", result)
        raise


if __name__ == "__main__":
    main(sys.argv[1])
