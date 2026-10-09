"""Direct official EmbodiedEvalRunner with scoped Ray environment propagation."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

# Frozen opendw_multigpu_owner.py names (called TOKEN/PHASE in that source).
OWNER_TOKEN_ENV = "OPENDW_SMOKE_OWNER_TOKEN"
OWNER_PHASE_ENV = "OPENDW_SMOKE_OWNER_PHASE"


def owner_markers(environment):
    markers = {key: environment.get(key) for key in (OWNER_TOKEN_ENV, OWNER_PHASE_ENV)}
    if any(not isinstance(value, str) or not value for value in markers.values()):
        raise ValueError("Native evaluation must inherit the exact owner token and phase")
    return markers


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--receipt-dir", type=Path, help="Runtime owner/phase directory; default config.parent")
    p.add_argument("--private-repo", required=True, type=Path)
    p.add_argument("--environment-fragment", required=True, type=Path)
    p.add_argument("--capture-dir", required=True, type=Path)
    p.add_argument("--namespace", required=True)
    p.add_argument("--ray-address", default="127.0.0.1:26379")
    args = p.parse_args()
    # Discovery CPU must see the real physical index4. Worker placement/scoped
    # runtime narrows GPU visibility; an external mask here renumbers devices.
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise ValueError("Do not mask the RLinf discovery driver")
    fragment = json.loads(args.environment_fragment.read_text())
    if "CUDA_VISIBLE_DEVICES" in fragment:
        raise ValueError("Do not mask the Ray job-level discovery processes")
    if os.environ.get("RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST") != fragment.get("RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST"):
        raise ValueError("Launch environment must apply the reviewed graphics scope before Python starts")
    if str(args.private_repo) not in fragment.get("PYTHONPATH", "").split(":"):
        raise ValueError("Scoped PYTHONPATH must name the private evaluation copy")
    fragment["WM_NATIVE_CAPTURE_DIR"] = str(args.capture_dir)
    fragment["WM_CAPTURE_POLICY"] = str(json.loads(args.config.read_text())["runner"].get("ckpt_path") or "original_pi05")
    # The Ray connection namespace and RLinf's manager lookup namespace must
    # agree. Overriding ray.init alone leaves Manager.get_runtime_env_vars()
    # exporting RLinf's default namespace to DeviceLockManager and workers.
    fragment["CLUSTER_NAMESPACE"] = args.namespace
    markers = owner_markers(os.environ)
    for key, value in markers.items():
        if key in fragment and fragment[key] != value:
            raise ValueError("Static environment fragment contains a stale owner marker")
    fragment.update(markers)
    receipt_dir = args.receipt_dir if args.receipt_dir is not None else args.config.parent
    receipt_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update(fragment)
    sys.path.insert(0, str(args.private_repo))
    import ray
    original_init = ray.init
    def scoped_init(*positional, **kwargs):
        kwargs["address"] = args.ray_address
        kwargs["namespace"] = args.namespace
        runtime = copy.deepcopy(kwargs.get("runtime_env") or {})
        runtime["env_vars"] = dict(runtime.get("env_vars") or {}, **fragment)
        kwargs["runtime_env"] = runtime
        result = original_init(*positional, **kwargs)
        context = ray.get_runtime_context()
        job = context.get_job_id()
        receipt = dict(driver_pid=os.getpid(), namespace=args.namespace,
            job_id=job.hex() if hasattr(job, "hex") else str(job), runtime_env_fragment=fragment)
        receipt_path = receipt_dir / "ray-job.json"
        if receipt_path.exists():
            previous = json.loads(receipt_path.read_text())
            if previous["namespace"] != receipt["namespace"] or previous["job_id"] != receipt["job_id"]:
                raise RuntimeError("Native evaluation attempted to overwrite another Ray job receipt")
        else:
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        return result
    ray.init = scoped_init
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    from omegaconf import OmegaConf
    from rlinf.config import validate_cfg
    from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
    from rlinf.scheduler import Cluster
    Cluster.NAMESPACE = args.namespace
    from rlinf.utils.placement import HybridComponentPlacement
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
    cfg = OmegaConf.load(args.config)
    if cfg.runner.task_type != "embodied_eval" or not cfg.runner.only_eval or "train" in cfg.env:
        raise ValueError("This launcher only accepts native evaluation without a train environment")
    cfg = validate_cfg(cfg)
    cluster = Cluster(cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path)
    placement = HybridComponentPlacement(cfg, cluster)
    rollout = MultiStepRolloutWorker.create_group(cfg).launch(cluster,
        name=cfg.rollout.group_name, placement_strategy=placement.get_strategy("rollout"))
    env = EnvWorker.create_group(cfg).launch(cluster,
        name=cfg.env.group_name, placement_strategy=placement.get_strategy("env"))
    runner = EmbodiedEvalRunner(cfg=cfg, rollout=rollout, env=env)
    runner.init_workers()
    runner.run()
    # WorkerGroup has no stop() API in this frozen RLinf revision. The existing
    # owner uses the recorded namespace/job plus process catalog for cleanup.
    ray.shutdown()


if __name__ == "__main__":
    main()
