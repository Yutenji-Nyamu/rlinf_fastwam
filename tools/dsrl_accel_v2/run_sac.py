# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Private Ray actor for discarded updates; publish only accepted runtime flags."""

import importlib.util
import json
import os
import signal
import time
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def main(directory: Any) -> None:
    """Run discarded SAC updates against an existing complete checkpoint."""
    d = Path(directory)
    p = json.loads((d / "plan.json").read_text())
    root = Path(p["training_source"])
    spec = importlib.util.spec_from_file_location("accepted_cleanup", p["legacy_ops"])
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)
    runtime = d / ("bench-runtime-" + str(os.getpid()))
    runtime.mkdir()
    namespace = p["namespace"] + "_bench_" + str(os.getpid())
    ops.RUN = runtime
    ops.ROOT = root
    ops.NAMESPACE = namespace
    from rlinf.config import validate_cfg
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement
    from tools.dsrl_accel_v2.sac_bench import BenchmarkPolicy

    Cluster.NAMESPACE = namespace
    cfg = validate_cfg(OmegaConf.load(d / "config/formal.yaml"))
    checkpoint = Path(cfg.runner.resume_dir)

    def stop(signum: int, *_: Any) -> None:
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
        actor.load_checkpoint(str(checkpoint / "actor")).wait()
        replay = checkpoint / "actor/sac_components/replay_buffer/rank_0"
        actor.load_benchmark_replay(str(replay)).wait()
        (d / "benchmark-input.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint),
                    "replay": str(replay),
                    "purpose": "discarded acceleration comparison; formal resumes original complete checkpoint",
                },
                indent=2,
            )
        )
        actor.benchmark(str(d / "sac-v2-timings.json")).wait()
        result = json.loads((d / "sac-v2-timings.json").read_text())
        assert result["complete"]
        formal = OmegaConf.load(d / "config/formal.yaml")
        for k, v in result["configuration"].items():
            formal.actor[k] = v
        OmegaConf.save(formal, d / "config/formal.yaml")
        (d / "selected-sac-v2.json").write_text(
            json.dumps(
                {
                    "selected": result["selected"],
                    "configuration": result["configuration"],
                    "speedup": result["sac_update_speedup"],
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
    import sys

    main(sys.argv[1])
