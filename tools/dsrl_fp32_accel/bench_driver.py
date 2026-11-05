# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Private Ray namespace, actor-only benchmark of actual checkpoint/replay."""

import importlib.util
import json
import os
import signal
import time
from pathlib import Path

from omegaconf import OmegaConf


def main(directory):
    d = Path(directory)
    p = json.loads((d / "plan.json").read_text())
    root = Path(p["training_source"])
    spec = importlib.util.spec_from_file_location("bench_cleanup", p["legacy_ops"])
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)
    runtime = d / ("bench-runtime-" + str(os.getpid()))
    runtime.mkdir()
    namespace = p["namespace"] + "_bench_" + str(os.getpid())
    ops.RUN = runtime
    ops.ROOT = root
    ops.NAMESPACE = namespace
    from tools.dsrl_fp32_accel.bench_worker import BenchmarkPolicy

    from rlinf.config import validate_cfg
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement

    Cluster.NAMESPACE = namespace
    cfg = OmegaConf.load(d / "config/smoke_resume.yaml")
    cfg.actor.model.openpi.dsrl_vectorized_q = False
    cfg.actor.model.openpi.dsrl_compile_q = False
    cfg.actor.defer_metrics = False
    cfg.actor.foreach_target_update = False
    cfg.actor.optim.fused = False
    cfg.actor.critic_optim.fused = False
    cfg.algorithm.entropy_tuning.optim.fused = False
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
        actor.load_checkpoint(str(Path(cfg.runner.resume_dir) / "actor")).wait()
        if p.get("benchmark_replay_path"):
            replay_path = Path(p["benchmark_replay_path"])
        else:
            prepared = json.loads((d.parent / "prepared-fp32.json").read_text())[
                "entries"
            ][str(p["gpu"])]
            oldowner = Path(prepared["old_owner"])
            oldplan = json.loads((oldowner / "plan.json").read_text())
            if "run" not in oldplan:
                oldplan = json.loads((oldowner.parent / "plan.json").read_text())
            checkpoints = Path(oldplan["run"]) / "formal/formal/checkpoints"
            candidates = [
                x
                for x in checkpoints.glob("global_step_*")
                if (x / "dojo-complete.json").is_file()
                and (x / "actor/sac_components/replay_buffer/rank_0").is_dir()
            ]
            reference = max(candidates, key=lambda x: int(x.name.rsplit("_", 1)[1]))
            replay_path = reference / "actor/sac_components/replay_buffer/rank_0"
        actor.load_benchmark_replay(str(replay_path)).wait()
        (d / "benchmark-input.json").write_text(
            json.dumps(
                {
                    "fp32_start_checkpoint": cfg.runner.resume_dir,
                    "real_replay": str(replay_path),
                    "purpose": "discarded timing experiment; formal run remains fresh",
                },
                indent=2,
            )
            + "\n"
        )
        actor.benchmark(str(d / "sac-timings.json")).wait()
        result = json.loads((d / "sac-timings.json").read_text())
        label = result["selected"]
        formal = OmegaConf.load(d / "config/formal.yaml")
        options = result["configuration"]
        formal.actor.defer_metrics = options["defer_metrics"]
        formal.actor.foreach_target_update = options["foreach_target_update"]
        formal.actor.optim.fused = options["fused"]
        formal.actor.critic_optim.fused = options["fused"]
        formal.algorithm.entropy_tuning.optim.fused = options["fused"]
        formal.actor.model.openpi.dsrl_vectorized_q = options["vectorized"]
        formal.actor.model.openpi.dsrl_compile_q = options["compiled"]
        formal.actor.model.openpi.dsrl_compile_q_mode = options["compile_mode"]
        formal.actor.model.openpi.dsrl_channels_last = options["channels_last"]
        OmegaConf.save(formal, d / "config/formal.yaml")
        (d / "selected-acceleration.json").write_text(
            json.dumps(
                {
                    "selected": label,
                    "sac_update_speedup": result["sac_update_speedup"],
                    "scope": "production SAC update from identical replay/state; whole-round speed pending",
                    "time": time.time(),
                },
                indent=2,
            )
            + "\n"
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
