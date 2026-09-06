#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-robotwin-rl
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle@fa8df6ed
HEAD=256eeeb4459b4bd5db85bfc6a0eb315771e8c38c
ROOT=/data/chenyiteng/results/rlinf-shenzhen/pi05
CONTROL_NAME=pi05-grpo-control-formal100-2gpu64x4-g8-b512-u5-m5-fixed32-eval5-phys45-localshard-v1
DVAC_NAME=pi05-grpo-dvac-action-adv-w0p5to1p5-formal100-2gpu64x4-g8-b512-u5-m5-fixed32-eval5-phys67-localshard-v1
CONTROL_RUN="$ROOT/runs/$CONTROL_NAME"
DVAC_RUN="$ROOT/runs/$DVAC_NAME"
CONTROL_PACKET="$ROOT/packets/$CONTROL_NAME"
DVAC_PACKET="$ROOT/packets/$DVAC_NAME"
CONTROL_EXPERIMENT=robotwin_pi05_grpo_control_formal100_2gpu64x4_g8_b512_u5_m5_fixed32_eval5_phys45_localshard_v1
DVAC_EXPERIMENT=robotwin_pi05_grpo_dvac_action_adv_w0p5to1p5_formal100_2gpu64x4_g8_b512_u5_m5_fixed32_eval5_phys67_localshard_v1

test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
test -s "$WT/examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_pi05.yaml"
test -s "$MODEL/model-00001-of-00003.safetensors"
test -s "$MODEL/model-00002-of-00003.safetensors"
test -s "$MODEL/model-00003-of-00003.safetensors"
test -s "$MODEL/physical-intelligence/robotwin/norm_stats.json"
test ! -e "$CONTROL_PACKET"; test ! -e "$DVAC_PACKET"
test ! -e "$CONTROL_RUN"; test ! -e "$DVAC_RUN"
RAY_ADDRESS=172.17.0.1:6389 "$VENV/bin/ray" status >/dev/null

source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment" RLINF_CODE_WORKING_DIR="$WT"
export PYTHONPATH="$WT:$ROBOTWIN" OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

common=(
  --config-path "$WT/examples/embodiment/config" --config-name robotwin_adjust_bottle_grpo_openpi_pi05
  runner.max_epochs=1000 runner.max_steps=100 runner.val_check_interval=5 runner.save_interval=10 runner.resume_dir=null
  algorithm.update_epoch=5 algorithm.group_size=8 algorithm.adv_type=grpo algorithm.loss_type=actor algorithm.filter_rewards=true
  env.train.total_num_envs=64 env.train.rollout_epoch=4 env.train.max_episode_steps=200 env.train.max_steps_per_rollout_epoch=200
  "env.train.assets_path=$ROBOTWIN" env.train.video_cfg.save_video=true
  env.eval.total_num_envs=32 env.eval.rollout_epoch=1 env.eval.max_episode_steps=200 env.eval.max_steps_per_rollout_epoch=200
  env.eval.use_fixed_reset_state_ids=true "env.eval.assets_path=$ROBOTWIN" env.eval.video_cfg.save_video=true
  actor.micro_batch_size=32 actor.global_batch_size=512 "actor.model.model_path=$MODEL" actor.model.num_steps=5
  ++actor.fsdp_config.checkpoint_format=local_shard
)
control=(
  'cluster.component_placement={actor\, env\, rollout:"4,5"}'
  "runner.logger.log_path=$CONTROL_RUN" "runner.logger.experiment_name=$CONTROL_EXPERIMENT"
  algorithm.logprob_type=chunk_level algorithm.dvac_gradient_weighting.mode=off
  algorithm.dvac_gradient_weighting.application=logprob_st
  algorithm.dvac_gradient_weighting.weight_min=null algorithm.dvac_gradient_weighting.weight_max=null
  "env.train.video_cfg.video_base_dir=$CONTROL_RUN/video/train" "env.train.task_config.save_path=$CONTROL_RUN/robotwin_data/train"
  "env.eval.video_cfg.video_base_dir=$CONTROL_RUN/video/eval" "env.eval.task_config.save_path=$CONTROL_RUN/robotwin_data/eval"
)
dvac=(
  'cluster.component_placement={actor\, env\, rollout:"6,7"}'
  "runner.logger.log_path=$DVAC_RUN" "runner.logger.experiment_name=$DVAC_EXPERIMENT"
  algorithm.logprob_type=action_level algorithm.dvac_gradient_weighting.mode=apply
  algorithm.dvac_gradient_weighting.application=action_advantage
  algorithm.dvac_gradient_weighting.selected_l=3 algorithm.dvac_gradient_weighting.warmup_steps=1
  algorithm.dvac_gradient_weighting.window_steps=5
  algorithm.dvac_gradient_weighting.weight_min=0.5 algorithm.dvac_gradient_weighting.weight_max=1.5
  "env.train.video_cfg.video_base_dir=$DVAC_RUN/video/train" "env.train.task_config.save_path=$DVAC_RUN/robotwin_data/train"
  "env.eval.video_cfg.video_base_dir=$DVAC_RUN/video/eval" "env.eval.task_config.save_path=$DVAC_RUN/robotwin_data/eval"
)

mkdir -p "$CONTROL_PACKET" "$DVAC_PACKET"
"$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${common[@]}" "${control[@]}" --cfg job --resolve > "$CONTROL_PACKET/resolved.yaml"
"$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${common[@]}" "${dvac[@]}" --cfg job --resolve > "$DVAC_PACKET/resolved.yaml"

"$VENV/bin/python" - "$CONTROL_PACKET/resolved.yaml" "$DVAC_PACKET/resolved.yaml" "$CONTROL_PACKET/contract.json" "$DVAC_PACKET/contract.json" <<'PY'
import json, sys, yaml

control = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
dvac = yaml.safe_load(open(sys.argv[2], encoding="utf-8"))
for cfg in (control, dvac):
    assert cfg["runner"]["max_steps"] == 100
    assert cfg["runner"]["val_check_interval"] == 5
    assert cfg["runner"]["save_interval"] == 10
    assert cfg["runner"]["resume_dir"] is None
    assert cfg["env"]["train"]["total_num_envs"] == 64
    assert cfg["env"]["train"]["rollout_epoch"] == 4
    assert cfg["env"]["eval"]["total_num_envs"] == 32
    assert cfg["env"]["eval"]["use_fixed_reset_state_ids"] is True
    assert cfg["algorithm"]["group_size"] == 8
    assert cfg["algorithm"]["adv_type"] == "grpo"
    assert cfg["algorithm"]["loss_type"] == "actor"
    assert cfg["algorithm"]["filter_rewards"] is True
    assert cfg["algorithm"]["update_epoch"] == 5
    assert cfg["actor"]["global_batch_size"] == 512
    assert cfg["actor"]["micro_batch_size"] == 32
    assert cfg["actor"]["model"]["model_path"].endswith("@fa8df6ed")
    assert cfg["actor"]["model"]["num_steps"] == 5
    assert cfg["actor"]["model"]["add_value_head"] is False
    assert cfg["actor"]["optim"]["lr"] == 5e-6
    assert cfg["actor"]["fsdp_config"]["checkpoint_format"] == "local_shard"
    assert cfg["env"]["enable_offload"] is True
    assert cfg["rollout"]["enable_offload"] is True
    assert cfg["actor"]["enable_offload"] is True
assert control["cluster"]["component_placement"] == {"actor, env, rollout": "4,5"}
assert dvac["cluster"]["component_placement"] == {"actor, env, rollout": "6,7"}
assert control["algorithm"]["logprob_type"] == "chunk_level"
assert control["algorithm"]["dvac_gradient_weighting"]["mode"] == "off"
w = dvac["algorithm"]["dvac_gradient_weighting"]
assert dvac["algorithm"]["logprob_type"] == "action_level"
assert (w["mode"], w["application"], w["selected_l"], w["warmup_steps"], w["window_steps"], w["weight_min"], w["weight_max"]) == ("apply", "action_advantage", 3, 1, 5, 0.5, 1.5)

def flatten(value, prefix=()):
    out = {}
    if isinstance(value, dict):
        for key, item in value.items(): out.update(flatten(item, prefix + (str(key),)))
    elif isinstance(value, list):
        for index, item in enumerate(value): out.update(flatten(item, prefix + (str(index),)))
    else: out[".".join(prefix)] = value
    return out

a, b = flatten(control), flatten(dvac)
diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
allowed = (
    "cluster.component_placement", "runner.logger.log_path", "runner.logger.experiment_name", "runner.output_dir",
    "env.train.video_cfg.video_base_dir", "env.train.task_config.save_path",
    "env.eval.video_cfg.video_base_dir", "env.eval.task_config.save_path",
    "algorithm.logprob_type", "algorithm.dvac_gradient_weighting",
)
unexpected = [key for key in diff if not key.startswith(allowed)]
assert not unexpected, unexpected
common = {
    "train_envs": 64, "rollout_epochs": 4, "trajectories_per_step": 256,
    "max_query_records": 1024, "eval_envs": 32, "eval_every": 5, "save_every": 10,
    "global_batch": 512, "micro_batch": 32, "update_epochs": 5,
    "optimizer_steps_per_outer": 10, "group_size": 8, "adv_type": "grpo",
    "loss_type": "actor", "denoise_steps": 5, "checkpoint_format": "local_shard",
}
payload = {"common": common, "allowed_pair_diff": diff, "unexpected": unexpected}
for path, kind in ((sys.argv[3], "pi05 GRPO Control"), (sys.argv[4], "pi05 GRPO-DVAC Action-Adv [0.5,1.5]")):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"kind": kind, **payload}, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
print(json.dumps({"diff_count": len(diff), "unexpected": unexpected, "diff": diff}, ensure_ascii=False, indent=2))
PY

printf '%q ' "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${common[@]}" "${control[@]}" > "$CONTROL_PACKET/command.txt"; printf '\n' >> "$CONTROL_PACKET/command.txt"
printf '%q ' "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${common[@]}" "${dvac[@]}" > "$DVAC_PACKET/command.txt"; printf '\n' >> "$DVAC_PACKET/command.txt"
printf '%s\n' "$HEAD" > "$CONTROL_PACKET/source_head.txt"
printf '%s\n' "$HEAD" > "$DVAC_PACKET/source_head.txt"
printf '%s\n' prepared > "$CONTROL_PACKET/packet_complete.txt"
printf '%s\n' prepared > "$DVAC_PACKET/packet_complete.txt"

echo "CONTROL_PACKET=$CONTROL_PACKET"
echo "DVAC_PACKET=$DVAC_PACKET"
echo SZ_PI05_GRPO_DUAL_FORMAL_PREPARED
