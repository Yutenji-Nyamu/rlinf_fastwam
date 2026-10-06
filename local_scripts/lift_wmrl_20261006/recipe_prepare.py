"""Derive lift_pot from the resolved working click_bell B16 recipe on CPU.

Keep policy, optimizer, parallelism and formal budget. Change task assets, the
frozen task-RM threshold, output identities and service endpoints only. This
preparer does not start/stop jobs or decide whether smoke passes.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from urllib.parse import urlparse


TASK = "lift_pot"
REWARD_SOURCE = "single_task_resnet_classifier"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flat(value, prefix=""):
    if not isinstance(value, dict):
        return {prefix: value}
    return {key: child for name, item in value.items()
            for key, child in flat(item, prefix + "." + name if prefix else name).items()}


def remote(value):
    path = PurePosixPath(value)
    if not path.is_relative_to("/data/chenyiteng") or ".." in path.parts:
        raise ValueError("Require an explicit absolute /data/chenyiteng path")
    return str(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def build(base, *, name, owner_dir, initial_state, native_seeds, sft_path, service_urls, threshold):
    require(bool(re.fullmatch("[A-Za-z0-9_-]+", name)), "Require unique simple run name")
    require(math.isfinite(threshold) and 0 < threshold <= 1, "Invalid frozen threshold")
    require(len(service_urls) == len(set(service_urls)) == 2, "Require two independent services")
    for endpoint in service_urls:
        parsed = urlparse(endpoint)
        require(parsed.scheme == "http" and parsed.hostname == "127.0.0.1" and parsed.port
                and not any((parsed.path, parsed.query, parsed.fragment, parsed.username, parsed.password)),
                "Services must be bare loopback HTTP host:port")
    cfg = copy.deepcopy(base)
    train, runner, actor, native = cfg["env"]["train"], cfg["runner"], cfg["actor"], cfg["env"]["eval"]
    require(cfg["cluster"]["component_placement"] == {"actor": "4,5", "env": "6,7", "rollout": "4,5"}, "Donor GPU placement differs")
    require((train["total_num_envs"], train["rollout_epoch"], train["group_size"], train["chunk"]) == (64, 8, 8, 32), "Donor rollout recipe differs")
    require(train["max_episode_steps"] == train["max_steps_per_rollout_epoch"] == 384, "Donor action budget differs")
    require(train["task_name"] == native["task_config"]["task_name"] == "click_bell", "Use the resolved working bell donor")
    require(train["reward_source"] == "worldarena_t5_classifier" and train["reward_mode"] == "first_success_binary", "Donor reward contract differs")
    require((actor["global_batch_size"], actor["micro_batch_size"], cfg["algorithm"]["update_epoch"]) == (2048, 8, 2), "Donor actor budget differs")
    require((runner["max_epochs"], runner["max_steps"], runner["save_interval"], runner["val_check_interval"]) == (1000, 200, 10, 10), "Donor formal schedule differs")
    require(actor["model"] == cfg["rollout"]["model"], "Actor and rollout policy differ")
    require(actor["model"]["model_path"] == sft_path and not re.search(r"global_step_|/checkpoint[s]?/", sft_path), "Start from the original SFT")
    require(native["total_num_envs"] == 32 and native["rollout_epoch"] == native["group_size"] == 1, "Donor native evaluation differs")
    require(train["auto_reset"] is False and train["ignore_terminations"] is False and train["use_rel_reward"] is False, "Donor first-success termination differs")
    require(native["max_episode_steps"] == native["max_steps_per_rollout_epoch"] == native["task_config"]["step_lim"] == 384, "Donor native C32/384 protocol differs")
    owner = PurePosixPath(remote(owner_dir))
    run = owner / "formal"
    train.update(task_name=TASK, initial_state_path=remote(initial_state), reward_source=REWARD_SOURCE,
                 success_reward_threshold=float(threshold), service_urls=service_urls)
    train["video_cfg"]["video_base_dir"] = str(run / "video/train")
    native["seeds_path"] = remote(native_seeds)
    native["task_config"].update(task_name=TASK, save_path=str(run / "robotwin_data/eval"))
    native["video_cfg"]["video_base_dir"] = str(run / "video/eval")
    runner.update(resume_dir=None, ckpt_path=None, per_worker_log_path=str(run / "worker-metrics"))
    runner["logger"].update(log_path=str(run), experiment_name=name + "-formal")
    cfg["env"]["group_name"] = "OpenDWEnv_" + name
    cfg["actor"]["group_name"] = "OpenDWActor_" + name
    cfg["rollout"]["group_name"] = "OpenDWRollout_" + name
    cfg["algorithm"]["dvac_gradient_weighting"]["output_dir"] = str(run / "unused-dv")
    smoke = copy.deepcopy(cfg)
    smoke["runner"].update(max_epochs=1, max_steps=1, save_interval=1, val_check_interval=1)
    smoke["env"]["train"].update(rollout_epoch=1, max_episode_steps=32, max_steps_per_rollout_epoch=32)
    smoke["env"]["eval"].update(max_episode_steps=32, max_steps_per_rollout_epoch=32)
    smoke["env"]["eval"]["task_config"]["step_lim"] = 32
    smoke["actor"]["global_batch_size"] = 64
    smoke = json.loads(json.dumps(smoke).replace(str(run), str(owner / "startup_smoke")).replace(name + "-formal", name + "-startup"))
    before, after = flat(base), flat(cfg)
    manifest = dict(task=TASK, initialization="original_SFT; smoke checkpoint is not used for formal",
        reward=dict(source=REWARD_SOURCE, frame_reduction="max8", threshold=threshold,
                    semantics="Preserve bell: max of eight predicted main-frame scores; give 1 once on threshold hit at C32 boundary; terminate row",
                    caveat="RM labels/threshold use native C32 boundary frames; internal generated-frame false positives need checking"),
        parallel=dict(N=64, G=8, wm_batch=16, actor_ranks=2, env_ranks=2, actor_microbatch=8),
        formal=dict(R=8, C=32, max_actions=384, trajectories_per_round=512, global_batch=2048,
                    update_epochs=2, rounds=200, save_eval_every=10),
        smoke=dict(R=1, C=32, max_actions=32, rounds=1, global_batch=64, native_eval_N=32,
                   tests_full_horizon_learning=False, note="Same parallel scale; one action block may contain no positive reward"),
        diff=[dict(key=key, before=before.get(key), after=after.get(key)) for key in sorted(before.keys() | after.keys()) if before.get(key) != after.get(key)],
        launched=False)
    return cfg, smoke, manifest


def native_baseline(formal, owner_dir, name):
    """Reuse the verified RM capture N16 path, with 32 distinct seed IDs."""
    cfg = copy.deepcopy(formal)
    runtime = PurePosixPath(remote(owner_dir)) / "native_baseline32"
    native = cfg["env"]["eval"]
    native.update(total_num_envs=16, rollout_epoch=2, use_fixed_reset_state_ids=False)
    native["video_cfg"].update(save_video=False, video_base_dir=str(runtime / "video"))
    native["task_config"].update(save_path=str(runtime / "robotwin_data"), eval_video_log=False)
    cfg["env"].pop("train")
    cfg["env"]["group_name"] = "LiftBaselineEnv_" + name
    cfg["rollout"]["group_name"] = "LiftBaselineRollout_" + name
    cfg["cluster"]["component_placement"] = {"env": "4", "rollout": "4"}
    cfg["runner"].update(task_type="embodied_eval", only_eval=True, resume_dir=None, ckpt_path=None,
        max_steps=1, max_epochs=1, val_check_interval=1, save_interval=-1,
        per_worker_log_path=str(runtime / "worker-metrics"))
    cfg["runner"]["logger"].update(log_path=str(runtime), experiment_name=name + "-native-baseline32")
    cfg["reward"]["use_reward_model"] = False
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--base-config-sha256", required=True)
    parser.add_argument("--reward-report", type=Path, required=True)
    parser.add_argument("--reward-checkpoint", type=Path, required=True)
    for key in ("name", "owner-dir", "initial-state", "native-seeds", "sft-path"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--service-urls", nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(sha(args.base_config) == args.base_config_sha256, "Donor config digest changed")
    reset = Path(args.initial_state)
    receipt = json.loads(reset.with_suffix(".json").read_text())
    require(receipt["task"] == TASK and receipt["count"] == 50 and receipt["sha256"] == sha(reset), "Require verified lift_pot clean50 reset")
    report = json.loads(args.reward_report.read_text())
    require(report["task_name"] == TASK and report["checkpoint_sha256"] == sha(args.reward_checkpoint), "RM task/checkpoint differs")
    threshold = report["validation"]["threshold"]
    require(threshold == report["test"]["threshold"], "Threshold was not frozen between validation and test")
    seeds = json.loads(Path(args.native_seeds).read_text()).get(TASK, {}).get("success_seeds")
    require(isinstance(seeds, list) and len(seeds) == len(set(seeds)) == 32 and all(type(seed) is int and seed >= 0 for seed in seeds), "Require exactly 32 frozen existing native seed IDs")
    cfg, smoke, manifest = build(json.loads(args.base_config.read_text()), name=args.name,
        owner_dir=args.owner_dir, initial_state=args.initial_state, native_seeds=args.native_seeds,
        sft_path=args.sft_path, service_urls=args.service_urls, threshold=threshold)
    manifest.update(base_config=str(args.base_config), base_sha256=sha(args.base_config),
        reward_report=str(args.reward_report), reward_report_sha256=sha(args.reward_report),
        reward_checkpoint=str(args.reward_checkpoint), reward_checkpoint_sha256=sha(args.reward_checkpoint),
        native_seeds_sha256=sha(args.native_seeds), reset_sha256=sha(reset),
        native_baseline=dict(N=16, R=2, total_requested_seeds=32, physical_gpu=4,
            fixed_reset_state_ids=False, actor_created=False, world_model_created=False,
            formal_evaluation="N32x1, same frozen 32 requested seeds; fixed IDs reused each evaluation"))
    baseline = native_baseline(cfg, args.owner_dir, args.name)
    args.output.mkdir(parents=True, exist_ok=False)
    for filename, value in (("formal.yaml", cfg), ("startup_smoke.yaml", smoke), ("native_baseline.json", baseline), ("config-contract.json", manifest)):
        (args.output / filename).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
