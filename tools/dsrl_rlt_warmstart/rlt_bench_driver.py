# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Reuse the accepted private-Ray benchmark lifecycle for the RLT worker."""

import importlib.util
import json
import os
import signal
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf


def main(directory):
    d = Path(directory)
    p = json.loads((d / "plan.json").read_text())
    root = Path(p["training_source"])
    spec = importlib.util.spec_from_file_location("rlt_bench_cleanup", p["legacy_ops"])
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)
    runtime = d / ("bench-runtime-" + str(os.getpid()))
    runtime.mkdir()
    namespace = p["namespace"] + "_bench_" + str(os.getpid())
    ops.RUN = runtime
    ops.ROOT = root
    ops.NAMESPACE = namespace
    from tools.dsrl_rlt_warmstart.rlt_bench_worker import BenchmarkPolicy

    from rlinf.config import validate_cfg
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement

    Cluster.NAMESPACE = namespace
    cfg = OmegaConf.load(d / "config/smoke_resume.yaml")
    cfg.actor.defer_metrics = False
    cfg.actor.foreach_target_update = False
    cfg.actor.optim.fused = False
    cfg.actor.critic_optim.fused = False
    cfg = validate_cfg(cfg)

    def stop(signum, *_):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    try:
        cluster = Cluster(
            cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
        )
        placement = HybridComponentPlacement(cfg, cluster).get_strategy("actor")
        actor = BenchmarkPolicy.create_group(cfg).launch(
            cluster, name=cfg.actor.group_name, placement_strategy=placement
        )
        actor.init_worker().wait()
        cp = (
            Path(p["run"]) / "smoke_resume/smoke_resume/checkpoints/global_step_2/actor"
        )
        actor.load_checkpoint(str(cp)).wait()
        actor.benchmark(str(d / "rlt-timings.json")).wait()
        result = json.loads((d / "rlt-timings.json").read_text())
        fast = result["selected"] == "combined"
        c = OmegaConf.load(d / "config/formal.yaml")
        c.actor.defer_metrics = fast
        c.actor.foreach_target_update = fast
        c.actor.optim.fused = fast
        c.actor.critic_optim.fused = fast
        OmegaConf.save(c, d / "config/formal.yaml")
        (d / "selected-acceleration.json").write_text(
            json.dumps(
                {
                    "selected": result["selected"],
                    "speedup": result["speedup"],
                    "scope": result["scope"],
                    "time": time.time(),
                },
                indent=2,
            )
        )
    finally:
        import ray

        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
        try:
            for attempt in range(3):
                try:
                    ops.cleanup_namespace(namespace, runtime / "owned-cleanup.json")
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(1)
        finally:
            ray.shutdown()


if __name__ == "__main__":
    main(sys.argv[1])
