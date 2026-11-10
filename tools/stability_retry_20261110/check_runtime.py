# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Exercise the actual config validator, C10 head and production scheduler on CPU."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from prepare import BASE, read, save


def main(method, gpu):
    torch.set_num_threads(2)
    entries = read(BASE / "prepared-dsrl.json")["entries"]
    root = next(e["source"] for e in entries.values() if e["method"] == method)
    sys.path.insert(0, root)
    from rlinf.config import validate_cfg
    from rlinf.scheduler import Cluster

    Cluster.NAMESPACE = "RLinf_retry_cpu_" + str(os.getpid())
    rows = {}
    for g, e in entries.items():
        if e["method"] != method or g != gpu:
            continue
        c = validate_cfg(OmegaConf.load(Path(e["deployment"]) / "config/formal.yaml"))
        row = {"config_validation": True, "effective_batch": c.actor.global_batch_size}
        if method == "rlt":
            from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy
            from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACFSDPPolicy

            m = RLTMLPPolicy(
                z_dim=2048,
                proprio_dim=14,
                action_dim=14,
                num_action_chunks=10,
                output_activation="identity",
            )
            obs = {
                "z_rl": torch.randn(512, 2048),
                "proprio": torch.randn(512, 14),
                "ref_chunk": torch.randn(512, 10, 14),
            }
            action, _, _ = m.sac_forward(obs)
            q = m.sac_q_forward(obs, action)
            loss = (
                q.square().mean()
                + (action.flatten(1) - obs["ref_chunk"].flatten(1)).square().mean()
            )
            loss.backward()
            assert action.numel() == 512 * 10 * 14 and q.shape[0] == 512
            assert all(
                p.grad is None or p.grad.isfinite().all() for p in m.parameters()
            )
            worker = object.__new__(RLTACFSDPPolicy)
            worker.cfg = c
            worker.rlt_schedule_cfg = c.algorithm.rlt_schedule
            worker.update_step = 0
            worker._warmup_ready_total_transitions = None
            worker._warmup_ready_total_episodes = None
            worker._freeze_rlt_dvac_baseline = lambda: None
            worker._rlt_dvac_baseline_metrics = dict
            counts = {
                "min_replay_size": 19999,
                "total_transitions_added": 19999,
                "total_episodes_added": 10,
                "transitions_since_train": 100,
            }
            worker._global_rlt_counters = lambda counts=counts: counts
            assert worker._rlt_updates_to_run()[0] == 0
            counts.update(min_replay_size=20000, total_transitions_added=20000)
            n, metrics = worker._rlt_updates_to_run()
            assert n == c.algorithm.rlt_schedule.max_updates_per_train_step
            worker.update_step = 30000
            counts.update(
                min_replay_size=20100,
                total_transitions_added=20100,
                transitions_since_train=100,
            )
            n, metrics = worker._rlt_updates_to_run()
            assert n == 500 and metrics["rlt/ready_for_online"] == 1
            row.update(
                action_shape=list(action.shape),
                finite_gradients=True,
                actual_scheduler_precollection=19999,
                actual_scheduler_post_warmup_updates=n,
            )
        rows[g] = row
    assert rows, (method, gpu)
    save(
        BASE / f"runtime-check-{method}-g{gpu}.json", {"result": "pass", "checks": rows}
    )
    print(json.dumps(rows))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
