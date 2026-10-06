"""Prepare bounded native reward-data collection; does not launch any process.

Adapted from rynn_binary_20261005/prepare_native_eval.py. The native rollout
policy and simulator settings are inherited; only the task, seed selection,
sample count, eval-only placement and output paths change. Native seed IDs
advance at each full-horizon auto-reset, so N16 x 8 requests 128 distinct IDs.
An internal simulator retry may use a different actual seed; the capture must
record or separately verify that distinction before claiming scene uniqueness.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re


NUM_ENVS = 16
HORIZON = 384
CHUNK = 32


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_config(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        import yaml
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a mapping")
    return value


def task_seeds(path, task, required=True):
    value = read_config(path)
    if task not in value:
        if required:
            raise ValueError(f"Seed file has no {task!r} entry: {path}")
        return None, []
    entry = value[task]
    seeds = entry.get("success_seeds") if isinstance(entry, dict) else None
    if (not isinstance(seeds, list) or not seeds
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError(f"Need a nonempty unique integer task.success_seeds list: {path}")
    return entry, seeds


def select_seeds(source, task, count, exclusions):
    entry, candidates = task_seeds(source, task)
    excluded = set()
    receipts = []
    for path, required in exclusions:
        _, seeds = task_seeds(path, task, required=required)
        excluded.update(seeds)
        receipts.append(dict(path=str(path), sha256=sha256(path), task_present=bool(seeds),
                             count=len(seeds), required=required))
    selected = [seed for seed in candidates if seed not in excluded][:count]
    if len(selected) != count:
        raise ValueError(f"Need {count} distinct unused seed IDs; only {len(selected)} remain")
    # Preserve the task entry's provenance rather than inventing a seed schema.
    output_entry = copy.deepcopy(entry)
    output_entry["success_seeds"] = selected
    return {task: output_entry}, receipts


def derive(original, output, task, seed_path, num_episodes, name):
    if not re.fullmatch(r"[a-z][a-z0-9_]*", task):
        raise ValueError("Task name must be an explicit RoboTwin task identifier")
    if num_episodes <= 0 or num_episodes % NUM_ENVS:
        raise ValueError("num-episodes must be a positive multiple of 16")
    cfg = copy.deepcopy(original)
    e = cfg["env"]["eval"]
    model = cfg["rollout"]["model"]
    if e["env_type"] != "robotwin" or not e.get("is_eval"):
        raise ValueError("Inherit an existing native RoboTwin evaluation config")
    if not isinstance(e.get("seeds_path"), str) or not e["seeds_path"]:
        raise ValueError("The existing config must declare env.eval.seeds_path")
    if (e["max_episode_steps"] != HORIZON or e["max_steps_per_rollout_epoch"] != HORIZON
            or e["task_config"]["step_lim"] != HORIZON):
        raise ValueError("Source config must already use the authorized 384-action horizon")
    if (model["num_action_chunks"] != CHUNK or model["openpi"]["action_chunk"] != CHUNK
            or model["openpi"]["num_images_in_input"] != 3
            or model["action_dim"] != 14 or model["openpi"]["action_horizon"] != 50):
        raise ValueError("Source policy must already be three-view / 14D / H50 / C32")
    if not e["auto_reset"] or not e["ignore_terminations"] or e["group_size"] != 1:
        raise ValueError("Require the existing full-horizon, ungrouped native evaluation protocol")
    if cfg["rollout"]["pipeline_stage_num"] != 1:
        raise ValueError("This preparation supports the existing single native rollout pipeline")
    if cfg["runner"].get("enable_decoupled_mode", False):
        raise ValueError("Inherit the existing coupled native evaluation runner")

    runtime = output / "run"
    e["task_config"]["task_name"] = task
    e["seeds_path"] = str(seed_path)
    e["total_num_envs"] = NUM_ENVS
    e["rollout_epoch"] = num_episodes // NUM_ENVS
    # With True every auto-reset repeats the same 16 seed IDs.
    e["use_fixed_reset_state_ids"] = False
    e["video_cfg"]["save_video"] = False
    e["video_cfg"]["video_base_dir"] = str(runtime / "video")
    e["task_config"]["save_path"] = str(runtime / "robotwin_data")
    e["task_config"]["eval_video_log"] = False
    cfg["env"].pop("train", None)
    cfg["env"]["group_name"] = "TaskRewardNativeEnv_" + name
    cfg["rollout"]["group_name"] = "TaskRewardNativeRollout_" + name
    cfg["cluster"]["component_placement"] = {"env": "4", "rollout": "4"}
    cfg["runner"].update(task_type="embodied_eval", only_eval=True, resume_dir=None,
        ckpt_path=None, max_steps=1, max_epochs=1, val_check_interval=1,
        save_interval=-1, per_worker_log_path=str(runtime / "worker-metrics"))
    cfg["runner"]["logger"].update(log_path=str(runtime), experiment_name=name)
    cfg["reward"]["use_reward_model"] = False
    return cfg


def flatten(value, prefix=""):
    if isinstance(value, dict):
        return {name: item for key, child in value.items()
                for name, item in flatten(child, prefix + "." + str(key) if prefix else str(key)).items()}
    return {prefix: value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "--formal-config", dest="config", required=True, type=Path)
    parser.add_argument("--task", "--task-name", dest="task", default="lift_pot")
    parser.add_argument("--seed-file", required=True, type=Path,
                        help="Existing task seed bank; task.success_seeds is preserved")
    parser.add_argument("--exclude-seed-file", action="append", default=[], type=Path,
                        help="Existing same-task tested/held-out seed bank; repeat as needed")
    parser.add_argument("--output", "--output-dir", dest="output", required=True, type=Path)
    parser.add_argument("--num-episodes", type=int, default=128)
    parser.add_argument("--name", help="Unique worker/log label; default output directory name")
    args = parser.parse_args()
    output = args.output.resolve()
    name = args.name or output.name
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", name):
        raise ValueError("Use a short alphanumeric, underscore or hyphen output/name label")
    original = read_config(args.config)
    inherited_seed_path = original["env"]["eval"].get("seeds_path")
    exclusions = [(path.resolve(), True) for path in args.exclude_seed_file]
    # Existing eval IDs are excluded automatically when their task entry exists.
    # A missing bank must be resolved by the caller; never silently assume it empty.
    if not isinstance(inherited_seed_path, str) or not inherited_seed_path:
        raise ValueError("Source config lacks env.eval.seeds_path")
    inherited = Path(inherited_seed_path)
    if not inherited.is_file():
        raise FileNotFoundError(f"Cannot check the inherited evaluation seed bank: {inherited}")
    if all(path != inherited.resolve() for path, _ in exclusions):
        exclusions.append((inherited.resolve(), False))
    seeds, excluded_receipts = select_seeds(args.seed_file, args.task, args.num_episodes, exclusions)
    seed_path = output / "native_seeds.json"
    cfg = derive(original, output, args.task, seed_path, args.num_episodes, name)
    before, after = flatten(original), flatten(cfg)
    changes = {key: {"before": before.get(key), "after": after.get(key)}
               for key in sorted(before.keys() | after.keys()) if before.get(key) != after.get(key)}
    selected = seeds[args.task]["success_seeds"]
    receipt = dict(schema_version=1, task=args.task, original_sft=True,
        source_config=str(args.config.resolve()), source_config_sha256=sha256(args.config),
        source_seed_file=str(args.seed_file.resolve()), source_seed_sha256=sha256(args.seed_file),
        inherited_eval_seed_file=inherited_seed_path, exclusions=excluded_receipts,
        selection="First unused IDs in the existing bank order; no policy/expert/outcome filtering",
        selected_seed_ids=selected, unique_seed_ids=len(set(selected)),
        seed_identity_scope="requested seed IDs; simulator fallback actual seeds require capture verification",
        native_episodes=args.num_episodes, environments=NUM_ENVS,
        rollout_epochs=args.num_episodes // NUM_ENVS, action_chunk=CHUNK, horizon=HORIZON,
        physical_gpus=[4], capture_mode="reward_native", creates_actor=False,
        creates_world_model=False, config_path=str(output / "native_eval.json"),
        seed_path=str(seed_path), runtime_output=str(output / "run"),
        capture_dir=str(output / "capture"), checkpoint=None,
        policy_model_path=cfg["rollout"]["model"]["model_path"],
        launch_arguments={"--config": str(output / "native_eval.json"),
            "--receipt-dir": str(output / "run"), "--capture-dir": str(output / "capture"),
            "--capture-mode": "reward_native"},
        launch_note="Use the existing scoped run_native_eval.py owner with its private-repo, environment-fragment and namespace; no launch is performed here",
        seed_note="RLinf shuffles and partitions this frozen bank using the inherited seed; auto-reset advances 16 IDs after each full horizon. The final reset wraps but executes no further episode. Verify 128 complete records (or num-episodes when overridden), requested IDs against this bank, actual seed/fallback evidence, and initial-frame duplicates before dataset conversion.",
        changes=changes)
    contents = {"native_seeds.json": seeds, "native_eval.json": cfg, "capture_plan.json": receipt}
    serialized = {key: (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
                  for key, value in contents.items()}
    if output.exists():
        # An exact preparation is idempotent. Different plans must use a fresh
        # output identity and never overwrite live captures or configuration.
        if not all((output / key).is_file() and (output / key).read_bytes() == value
                   for key, value in serialized.items()):
            raise ValueError("Output already exists with different/incomplete preparation; use a fresh unique path")
        status = "already_prepared"
    else:
        output.mkdir(parents=True, exist_ok=False)
        for key, value in serialized.items():
            with (output / key).open("xb") as stream:
                stream.write(value)
        status = "prepared"
    print(json.dumps(dict(status=status, task=args.task, native_episodes=args.num_episodes,
        unique_seed_ids=len(set(selected)), physical_gpus=[4], capture_mode="reward_native",
        output=str(output), plan_sha256=sha256(output / "capture_plan.json"),
        training_or_collection_started=False)))


if __name__ == "__main__":
    main()
