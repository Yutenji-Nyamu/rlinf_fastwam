"""Build a one-iteration OpenDW GRPO config from the pinned Sidney Control.

This script writes reviewable configuration only. It does not launch Ray, load
models, reserve GPUs, or touch a running experiment. JSON is also valid YAML.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from urllib.parse import urlparse


BASE_REF = "2151a08ee1bd75df1bef0d8190e594bd5c7f7977"
BASE_CONFIG = "examples/embodiment/config/sz2_can256_clean_resume_n32-20260927-v1.yaml"
ALLOWED_GPUS = {4, 5, 6, 7}


def parse_gpus(value: str) -> list[int]:
    items = [int(part) for part in value.split(",")]
    if not items or len(items) != len(set(items)) or not set(items) <= ALLOWED_GPUS:
        raise ValueError("Each GPU list must contain unique physical GPU IDs in 4-7")
    return items


def flatten(value, prefix=""):
    if not isinstance(value, dict):
        return {prefix: value}
    return {key: item for name, child in value.items()
            for key, item in flatten(child, f"{prefix}.{name}" if prefix else name).items()}


def build(args):
    repo = args.repo.resolve()
    # The per-command exception applies only to the user-selected checkout.
    original = subprocess.check_output(
        ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo),
         "show", f"{BASE_REF}:{BASE_CONFIG}"])
    base = json.loads(original)
    cfg = copy.deepcopy(base)
    n = args.num_envs
    actor_gpus = parse_gpus(args.actor_gpus)
    env_gpus = parse_gpus(args.env_gpus)
    rollout_gpus = parse_gpus(args.rollout_gpus)
    if n < 8 or n % (8 * len(env_gpus)):
        raise ValueError("N must give each EnvWorker a whole G8 group")
    if n % len(actor_gpus) or n % len(rollout_gpus):
        raise ValueError("N must be divisible by actor and rollout world sizes")
    if args.micro_batch_size <= 0 or n % (args.micro_batch_size * len(actor_gpus)):
        raise ValueError("Global batch N must divide microbatch * actor world size")
    if args.update_epochs not in (1, 2):
        raise ValueError("Smoke update_epochs must be 1 or inherited 2")
    for remote_path in (args.run_dir, args.initial_state, args.policy_path):
        if not remote_path.is_absolute() or not str(remote_path).startswith("/data/chenyiteng/"):
            raise ValueError("Remote model, reset and output paths must be below /data/chenyiteng")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.name):
        raise ValueError("Use a simple unique experiment name")
    url = urlparse(args.service_url)
    if url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost") or not url.port:
        raise ValueError("The smoke service must use an explicit local HTTP port")

    cfg["hydra"] = {"run": {"dir": "."}, "output_subdir": None}
    cfg["cluster"]["component_placement"] = {
        "actor": ",".join(map(str, actor_gpus)),
        "env": ",".join(map(str, env_gpus)),
        "rollout": ",".join(map(str, rollout_gpus)),
    }
    runner = cfg["runner"]
    runner.update(max_epochs=1, max_steps=1, only_eval=False,
                  val_check_interval=-1, save_interval=1,
                  resume_dir=None, ckpt_path=None, overlap_env_bootstrap=False,
                  per_worker_log=True, per_worker_log_path=str(args.run_dir / "worker-metrics"))
    runner["logger"].update(log_path=str(args.run_dir), experiment_name=args.name)

    # This smoke has one 32-action transition, not a full 400-action evaluation.
    cfg["env"] = {
        "group_name": "OpenDWEnv_" + args.name,
        "enable_offload": True,
        "train": {
            "env_type": "opendw_robotwin", "wm_env_type": "robotwin",
            "task_name": "adjust_bottle", "total_num_envs": n,
            "rollout_epoch": 1, "group_size": 8,
            "max_episode_steps": 32, "max_steps_per_rollout_epoch": 32,
            "chunk": 32, "frame_stride": 4, "image_size": [256, 256],
            "seed": 0, "auto_reset": False, "ignore_terminations": False,
            "is_eval": False, "center_crop": False,
            "use_rel_reward": True, "reward_coef": 1.0,
            "success_reward_threshold": 0.9, "use_fixed_reset_state_ids": True,
            "initial_state_path": str(args.initial_state),
            "service_url": args.service_url, "request_timeout_s": args.request_timeout_s,
            "enable_offload": True, "enable_init_offload": True,
            "video_cfg": {"save_video": False, "info_on_video": False,
                          "video_base_dir": str(args.run_dir / "video")},
        },
    }
    cfg["algorithm"]["update_epoch"] = args.update_epochs
    # Latest Clean uses observe-only DV telemetry. Pure smoke needs no DV files;
    # its all-one weights and the GRPO objective remain unchanged.
    cfg["algorithm"]["dvac_gradient_weighting"]["mode"] = "off"
    cfg["algorithm"]["dvac_gradient_weighting"]["save_step_tensors"] = False
    cfg["algorithm"]["dvac_gradient_weighting"]["output_dir"] = str(args.run_dir / "unused-dv")
    actor = cfg["actor"]
    actor.update(group_name="OpenDWActor_" + args.name,
                 global_batch_size=n, micro_batch_size=args.micro_batch_size,
                 enable_offload=True)
    actor["model"]["model_path"] = str(args.policy_path)
    actor["model"]["num_action_chunks"] = 32
    actor["model"]["openpi"].update(action_chunk=32, action_horizon=50)
    # Explicit shared model contract prevents train/eval or actor/rollout drift.
    cfg["rollout"]["model"] = copy.deepcopy(actor["model"])
    cfg["rollout"].update(group_name="OpenDWRollout_" + args.name,
                          pipeline_stage_num=1, enable_offload=True)
    before, after = flatten(base), flatten(cfg)
    diff = [{"key": key, "before": before.get(key), "after": after.get(key)}
            for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)]
    manifest = {
        "source_ref": BASE_REF, "source_config": BASE_CONFIG,
        "source_config_sha256": hashlib.sha256(original).hexdigest(),
        "writes_configuration_only": True, "launched": False,
        "physical_placement": cfg["cluster"]["component_placement"],
        "contract": {"task": "adjust_bottle", "N": n, "G": 8, "R": 1,
                     "L": 32, "C": 32, "H": 50, "M": 10,
                     "runner_iterations": 1, "update_epochs": args.update_epochs,
                     "global_batch_size": n, "micro_batch_size": args.micro_batch_size,
                     "actor_world_size": len(actor_gpus),
                     "gradient_accumulation": n // (args.micro_batch_size * len(actor_gpus)),
                     "scheduled_optimizer_steps": args.update_epochs,
                     "native_evaluation": False, "effective_gradient_not_guaranteed": True},
        "config_diff": diff,
    }
    return cfg, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--run-dir", type=PurePosixPath, required=True)
    parser.add_argument("--initial-state", type=PurePosixPath, required=True)
    parser.add_argument("--policy-path", type=PurePosixPath, required=True)
    parser.add_argument("--service-url", required=True)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--actor-gpus", default="4")
    parser.add_argument("--env-gpus", default="4")
    parser.add_argument("--rollout-gpus", default="4")
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--request-timeout-s", type=int, default=7200)
    args = parser.parse_args()
    if args.request_timeout_s <= 0:
        parser.error("request timeout must be positive")
    cfg, manifest = build(args)
    args.output.mkdir(parents=True, exist_ok=False)
    config_path = args.output / (args.name + ".yaml")
    config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest["output_config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    (args.output / "config-contract.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"config": str(config_path), "contract": manifest["contract"], "launched": False}))


if __name__ == "__main__":
    main()
