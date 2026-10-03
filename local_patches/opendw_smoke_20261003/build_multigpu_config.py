"""Write the reviewed N64/G8/R8 four-GPU OpenDW smoke configuration.

Configuration generation only: no model, Ray, HTTP request, or GPU execution.
The pinned Control supplies policy/algorithm/optimizer settings. This entry
supports one runner iteration at L32 or L384; formal training is not exposed.
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
PLACEMENT = {"actor": "4,5", "env": "6,7", "rollout": "4,5"}


def flatten(value, prefix=""):
    if not isinstance(value, dict):
        return {prefix: value}
    return {key: item for name, child in value.items()
            for key, item in flatten(child, f"{prefix}.{name}" if prefix else name).items()}


def normalized_service_urls(values):
    if len(values) != 2:
        raise ValueError("Exactly two ordered service URLs are required for env ranks 0/1")
    result, endpoints = [], []
    for value in values:
        parsed = urlparse(value)
        if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost")
                or parsed.port is None or not 1 <= parsed.port <= 65535
                or parsed.username is not None or parsed.password is not None
                or parsed.path not in ("", "/") or parsed.params or parsed.query or parsed.fragment):
            raise ValueError("Services must be loopback HTTP URLs with an explicit port and no path/auth/query")
        result.append(f"http://{parsed.hostname}:{parsed.port}")
        # localhost and 127.0.0.1 identify the same local endpoint for this run.
        endpoints.append(parsed.port)
    if len(set(endpoints)) != 2:
        raise ValueError("GPU6 and GPU7 must use different local service ports")
    return result


def require_remote_path(value):
    path = PurePosixPath(value)
    root = PurePosixPath("/data/chenyiteng")
    if not path.is_absolute() or not path.is_relative_to(root) or ".." in path.parts or path == root:
        raise ValueError("Remote model, reset and output paths must be strictly below /data/chenyiteng")
    return path


def build(args):
    repo = args.repo.resolve()
    original = subprocess.check_output([
        "git", "-c", "gc.auto=0", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo),
        "show", f"{BASE_REF}:{BASE_CONFIG}",
    ])
    base = json.loads(original)
    cfg = copy.deepcopy(base)
    run_dir, initial_state, policy_path = map(require_remote_path, (args.run_dir, args.initial_state, args.policy_path))
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.name):
        raise ValueError("Use a simple unique experiment name")
    if args.request_timeout_s <= 0:
        raise ValueError("Request timeout must be positive")
    service_urls = normalized_service_urls(args.service_urls)
    length = getattr(args, 'episode_steps', 32)
    if length not in (32, 384):
        raise ValueError('Reviewed smoke lengths are exactly 32 and 384')
    global_batch = 512 if length == 32 else 2048
    control_train = base["env"]["train"]
    assert control_train["rollout_epoch"] == 8
    assert control_train["group_size"] == base["algorithm"]["group_size"] == 8
    assert control_train["use_fixed_reset_state_ids"] is False
    assert base["algorithm"]["update_epoch"] == 2
    assert base["actor"]["model"]["openpi"]["num_steps"] == 10
    assert base["actor"]["model"]["openpi"]["num_images_in_input"] == 3

    cfg["hydra"] = {"run": {"dir": "."}, "output_subdir": None}
    cfg["cluster"]["component_placement"] = copy.deepcopy(PLACEMENT)
    cfg["runner"].update(
        max_epochs=1, max_steps=1, only_eval=False, val_check_interval=-1, save_interval=1,
        resume_dir=None, ckpt_path=None, overlap_env_bootstrap=False,
        per_worker_log=True, per_worker_log_path=str(run_dir / "worker-metrics"),
    )
    cfg["runner"]["logger"].update(log_path=str(run_dir), experiment_name=args.name)
    cfg["env"] = {
        "group_name": "OpenDWEnv_" + args.name,
        "enable_offload": True,
        "train": {
            "env_type": "opendw_robotwin", "wm_env_type": "robotwin", "task_name": "adjust_bottle",
            "total_num_envs": 64, "rollout_epoch": control_train["rollout_epoch"], "group_size": 8,
            "max_episode_steps": length, "max_steps_per_rollout_epoch": length,
            "chunk": 32, "frame_stride": 4, "image_size": [256, 256],
            "seed": control_train["seed"], "auto_reset": False, "ignore_terminations": False,
            "is_eval": False, "center_crop": control_train["center_crop"],
            "use_rel_reward": control_train["use_rel_reward"], "reward_coef": control_train["reward_coef"],
            "success_reward_threshold": 0.9,
            "use_fixed_reset_state_ids": control_train["use_fixed_reset_state_ids"],
            "initial_state_path": str(initial_state), "service_urls": service_urls,
            "request_timeout_s": args.request_timeout_s,
            "enable_offload": True, "enable_init_offload": True,
            "video_cfg": {"save_video": False, "info_on_video": False,
                          "video_base_dir": str(run_dir / "video")},
        },
    }
    # Match the existing pure-GRPO smoke: no observe-only DV tensor output.
    cfg["algorithm"]["dvac_gradient_weighting"].update(
        mode="off", save_step_tensors=False, output_dir=str(run_dir / "unused-dv"),
    )
    actor = cfg["actor"]
    actor.update(group_name="OpenDWActor_" + args.name, global_batch_size=global_batch,
                 micro_batch_size=8, enable_offload=True)
    actor["model"]["model_path"] = str(policy_path)
    actor["model"]["num_action_chunks"] = 32
    actor["model"]["openpi"].update(action_chunk=32, action_horizon=50)
    cfg["rollout"]["model"] = copy.deepcopy(actor["model"])
    cfg["rollout"].update(group_name="OpenDWRollout_" + args.name,
                          pipeline_stage_num=1, enable_offload=True)

    n, r, g, chunk = 64, 8, 8, 32
    actor_world, env_world, gb, micro, updates = 2, 2, global_batch, 8, 2
    trajectories = n * r
    chunks = trajectories * (length // chunk)
    accumulation = gb // (micro * actor_world)
    optimizer_steps = chunks // gb * updates
    assert chunks % gb == 0 and gb % (micro * actor_world) == 0
    assert (optimizer_steps, accumulation) == ((2, 32) if length == 32 else (6, 128))
    assert cfg["actor"]["model"] == cfg["rollout"]["model"]
    assert cfg["algorithm"]["filter_rewards"] is True
    assert cfg["algorithm"]["rewards_lower_bound"] == 0.1
    assert cfg["algorithm"]["rewards_upper_bound"] == 0.9
    assert cfg["env"]["train"]["reward_coef"] == 1.0
    before, after = flatten(base), flatten(cfg)
    manifest = {
        "source_ref": BASE_REF, "source_config": BASE_CONFIG,
        "source_config_sha256": hashlib.sha256(original).hexdigest(),
        "writes_configuration_only": True, "launched": False,
        "physical_placement": copy.deepcopy(PLACEMENT),
        "contract": {
            "mode": "multigpu_short_smoke" if length == 32 else "multigpu_signal_smoke", "task": "adjust_bottle",
            "N": n, "G": g, "R": r, "L": length, "C": chunk, "H": 50, "M": 10,
            "runner_iterations": 1, "trajectories_per_iteration": trajectories,
            "chunks_per_trajectory": length // chunk, "chunk_samples_per_iteration": chunks,
            "grpo_groups_per_iteration": trajectories // g,
            "global_batch_size": gb, "micro_batch_size": micro, "update_epochs": updates,
            "actor_world_size": actor_world, "rollout_world_size": 2, "env_world_size": env_world,
            "pipeline_stage_num": 1, "environments_per_env_rank": n // env_world,
            "gradient_accumulation_per_actor": accumulation,
            "scheduled_optimizer_steps": optimizer_steps,
            "micro_forward_backward_calls_per_actor": accumulation * optimizer_steps,
            "planned_wm_rows_max": chunks, "planned_environment_actions_max": trajectories * length,
            "planned_future_frames_max": chunks * 8,
            "reset_sampling": "Each env rank resamples four G8 starts with replacement after every rollout epoch",
            "expected_reset_count": 50, "reset_asset_contents_verified_by_generator": False,
            "native_evaluation": False, "save_interval": 1, "val_check_interval": -1,
            "effective_gradient_not_guaranteed": True,
        },
        "required_service_launch_contract": [
            {"env_rank": rank, "physical_gpu": 6 + rank, "url": service_urls[rank],
             "model_batch_size": 1, "logical_environments": 32, "wm_rows_max": chunks // env_world}
            for rank in range(env_world)
        ],
        "config_diff": [{"key": key, "before": before.get(key), "after": after.get(key)}
                        for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)],
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
    parser.add_argument("--service-urls", nargs=2, required=True, metavar=("GPU6_URL", "GPU7_URL"))
    parser.add_argument("--request-timeout-s", type=int, default=7200)
    parser.add_argument("--episode-steps", type=int, choices=(32, 384), default=32)
    args = parser.parse_args()
    cfg, manifest = build(args)
    args.output.mkdir(parents=True, exist_ok=False)
    config_path = args.output / (args.name + ".yaml")
    config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest["output_config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    (args.output / "config-contract.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"config": str(config_path), "contract": manifest["contract"], "launched": False}))


if __name__ == "__main__":
    main()
