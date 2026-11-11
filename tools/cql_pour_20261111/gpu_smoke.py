"""Eight production updates; checkpoint/replay are read only and results discarded."""

import importlib.util
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf


def main(directory):
    d = Path(directory)
    p = json.loads((d / "plan.json").read_text())
    spec = importlib.util.spec_from_file_location("cql_cleanup", p["legacy_ops"])
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)
    runtime = d / ("smoke-runtime-" + str(os.getpid()))
    runtime.mkdir()
    namespace = p["namespace"] + "_cql_smoke"
    ops.RUN, ops.ROOT, ops.NAMESPACE = runtime, Path(p["training_source"]), namespace
    from rlinf.config import validate_cfg
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement
    from tools.cql_pour_20261111.smoke_worker import CQLSmokePolicy

    Cluster.NAMESPACE = namespace
    cfg = validate_cfg(OmegaConf.load(d / "config/formal.yaml"))

    def stop(signum, *_):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    try:
        cluster = Cluster(cluster_cfg=cfg.cluster, distributed_log_dir=None)
        placement = HybridComponentPlacement(cfg, cluster).get_strategy("actor")
        actor = CQLSmokePolicy.create_group(cfg).launch(
            cluster, name=cfg.actor.group_name, placement_strategy=placement
        )
        actor.init_worker().wait()
        actor.load_checkpoint(str(Path(cfg.runner.resume_dir) / "actor")).wait()
        actor.load_benchmark_replay(
            str(
                Path(cfg.runner.resume_dir)
                / "actor/sac_components/replay_buffer/rank_0"
            )
        ).wait()
        actor.cql_smoke(str(d / "gpu-smoke.json")).wait()
        result = json.loads((d / "gpu-smoke.json").read_text())
        assert result["passed"] and result["updates"] == 8
        assert all(math.isfinite(v) for v in result["metrics"].values())
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
