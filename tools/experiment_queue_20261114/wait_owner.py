# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Reserve one slot while waiting, then reuse its existing experiment owner."""

import fcntl
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path


def checkpoint_ready(path, method, require_alpha=False):
    """Recognize the distinct full-state DSRL and RLT checkpoint layouts."""
    p = Path(path)
    required = [
        "dojo-complete.json",
        "actor/sac_components/target_model/checkpoint_rank_0.pt",
    ]
    if method == "dsrl":
        required += [
            "actor/local_shard_checkpoint/checkpoint_rank_0.pt",
            "actor/sac_components/dsrl_trainer_state_rank_0.pt",
            "actor/sac_components/replay_buffer/rank_0/dsrl_transition_replay.pt",
        ]
        shard = (
            p / "actor/sac_components/alpha/dcp_checkpoint" if require_alpha else None
        )
    elif method == "rlt":
        required += [
            "actor/dcp_checkpoint/.metadata",
            "actor/model_state_dict/full_weights.pt",
            "actor/sac_components/rlt_trainer_state/checkpoint_rank_0.pt",
            "actor/sac_components/rlt_trainer_state/complete.json",
            "actor/sac_components/replay_buffer/rank_0/metadata.json",
        ]
        shard = p / "actor/dcp_checkpoint"
    else:
        raise ValueError(method)
    return all((p / n).is_file() and (p / n).stat().st_size for n in required) and (
        shard is None
        or ((shard / ".metadata").is_file() and any(shard.glob("*.distcp")))
    )


def waiting_finished(blockers, identity_alive, release_exists):
    """Every predecessor must have exited and completed its cleanup."""
    return all(
        not identity_alive(b["identity"]) and release_exists(b["release"])
        for b in blockers
    )


def main(directory):
    d = Path(directory)
    wait = json.loads((d / "wait.json").read_text())
    sys.path.insert(0, wait["queue_ops"])
    from common import proc, read, same, save

    p = read(d / "plan.json")
    assert os.getuid() == wait["uid"] and p["gpu"] in (4, 5, 6, 7)
    me = proc(os.getpid())
    save(d / "identity.json", me)
    save(d / "status.json", dict(phase=wait["phase"], time=time.time()))

    def stop(*_):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, stop)
    try:
        while not (d / "registered.json").exists():
            time.sleep(0.2)
        assert read(d / "registered.json")["identity"] == me
        while not waiting_finished(wait["blockers"], same, lambda f: Path(f).is_file()):
            time.sleep(2)
        if wait["kind"] == "continuation":
            from omegaconf import OmegaConf

            cfg = OmegaConf.load(d / "config/formal.yaml")
            require_alpha = cfg.algorithm.entropy_tuning.alpha_type != "fixed_alpha"
            # Prefer the planned segment endpoint. A failed segment can still
            # continue from its latest complete full-state save.
            checkpoint = Path(wait["checkpoint"])
            if not checkpoint_ready(checkpoint, p["method"], require_alpha):
                candidates = [
                    cp
                    for cp in checkpoint.parent.glob("global_step_*")
                    if checkpoint_ready(cp, p["method"], require_alpha)
                ]
                assert candidates, "No complete checkpoint; release to next RLT"
                checkpoint = max(
                    candidates, key=lambda x: int(x.name.rsplit("_", 1)[1])
                )
            step = read(checkpoint / "dojo-complete.json")["step"]
            assert 0 < step < wait["target"]
            cfg.runner.resume_dir = str(checkpoint)
            OmegaConf.save(cfg, d / "config/formal.yaml")
            save(
                d / "continuation-ready.json",
                dict(checkpoint=str(checkpoint), step=step),
            )

        active_path = Path(wait["active_index"])
        with active_path.with_suffix(".continuation.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            active = read(active_path)
            slot = active["slots"][str(p["gpu"])]
            assert slot["owner"] in wait["allowed_previous_owners"]
            slot.update(
                deployment=str(d),
                owner=str(d),
                identity=me,
                source=p["training_source"],
                run=p["run"],
                bench=p["bench"],
                method=p["method"],
                task=p["task"],
                total_target_steps=wait["target"],
            )
            slot.pop("pending_continuation", None)
            if wait["kind"] == "rlt":
                slot.pop("pending_rlt", None)
            active["updated"] = time.time()
            save(active_path, active)

        # Import only after selecting the source: each waiter starts with an
        # empty CUDA mask and uses one immutable source tree for its lifetime.
        sys.path.insert(0, p["training_source"])
        if p["bench"] == "dojo" and p["method"] == "rlt":
            from tools.dsrl_rlt_warmstart.rlt_dojo_owner import main as owner
        elif p["bench"] == "dojo":
            from tools.dsrl_accel_v2.dojo_owner import main as owner
        else:
            from tools.dsrl_rlt_warmstart.rt2_owner import main as owner
        owner(str(d))
    except BaseException:
        if not (d / "released.json").exists():
            result = dict(
                phase="FAILED_RETURN_RLT",
                time=time.time(),
                error=traceback.format_exc(),
            )
            save(d / "status.json", result)
            save(d / "released.json", result)
        raise


if __name__ == "__main__":
    main(sys.argv[1])
