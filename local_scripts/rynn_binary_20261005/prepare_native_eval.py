"""Derive native N16x2 evaluation from the resolved formal config, no WM/actor."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def derive(original, output, checkpoint, name):
    cfg = copy.deepcopy(original)
    e = cfg["env"]["eval"]
    assert e["env_type"] == "robotwin" and e["task_config"]["task_name"] in ("adjust_bottle", "click_bell")
    assert e["max_episode_steps"] == e["max_steps_per_rollout_epoch"] == 384
    assert cfg["rollout"]["model"]["num_action_chunks"] == 32
    assert cfg["rollout"]["model"]["openpi"]["num_images_in_input"] == 3
    e["total_num_envs"], e["rollout_epoch"] = 16, 2
    e["video_cfg"]["save_video"] = False
    e["video_cfg"]["video_base_dir"] = str(output / "video")
    e["task_config"]["save_path"] = str(output / "robotwin_data")
    # Raw sidecar images are all that this bounded classifier diagnostic needs.
    e["task_config"]["eval_video_log"] = False
    cfg["env"].pop("train", None)
    cfg["env"]["group_name"] = "BinaryNativeEnv_" + name
    cfg["rollout"]["group_name"] = "BinaryNativeRollout_" + name
    cfg["cluster"]["component_placement"] = {"env": "4", "rollout": "4"}
    r = cfg["runner"]
    r.update(task_type="embodied_eval", only_eval=True, resume_dir=None, ckpt_path=str(checkpoint) if checkpoint else None,
             max_steps=1, max_epochs=1, val_check_interval=1, save_interval=-1,
             per_worker_log_path=str(output / "worker-metrics"))
    r["logger"].update(log_path=str(output), experiment_name=name)
    cfg["reward"]["use_reward_model"] = False
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--formal-config", required=True, type=Path)
    weights = p.add_mutually_exclusive_group(required=True)
    weights.add_argument("--checkpoint-file", type=Path)
    weights.add_argument("--original-sft", action="store_true")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--config-dir", type=Path, help="Freeze config here without creating output-dir; default output-dir")
    p.add_argument("--name", default="rynn-native-binary32-v1")
    args = p.parse_args()
    if args.checkpoint_file is not None and not args.checkpoint_file.is_file():
        raise ValueError("Need an exported full policy state_dict, not a sharded actor directory")
    config_dir = args.config_dir if args.config_dir is not None else args.output_dir
    if any((config_dir / name).exists() for name in ("native_eval.json", "derivation.json")):
        raise ValueError("Prepared evaluation configuration already exists")
    if config_dir == args.output_dir and args.output_dir.exists():
        raise ValueError("Evaluation output directory must be fresh")
    config_dir.mkdir(parents=True, exist_ok=True)
    raw = args.formal_config.read_bytes()
    cfg = derive(json.loads(raw), args.output_dir, args.checkpoint_file, args.name)
    (config_dir / "native_eval.json").write_text(json.dumps(cfg, indent=2) + "\n")
    receipt = dict(formal_config=str(args.formal_config), formal_sha256=hashlib.sha256(raw).hexdigest(),
        checkpoint=str(args.checkpoint_file) if args.checkpoint_file else None,
        original_sft=args.original_sft, task=cfg["env"]["eval"]["task_config"]["task_name"],
        checkpoint_bytes=args.checkpoint_file.stat().st_size if args.checkpoint_file else None,
        native_episodes=32, environments=16, rollout_epochs=2, action_chunk=32, horizon=384,
        physical_gpus=[4], creates_actor=False, creates_world_model=False,
        changes=["eval-only rollout/env", "N16x2 on GPU4", "per-episode raw sidecar replaces mosaic video", "new output paths"])
    receipt.update(config_path=str(config_dir / "native_eval.json"), runtime_output=str(args.output_dir),
                   output_dir_created_during_preparation=config_dir == args.output_dir)
    (config_dir / "derivation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
