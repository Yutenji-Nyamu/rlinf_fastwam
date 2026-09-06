#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-dvac-action-adv-fix
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
HEAD=74617ced87d64045ab6850d0efd90956a494af66
ROOT=/data/chenyiteng/results/rlinf-shenzhen/ppo
CONTROL_NAME=ppo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-localshard-v1
DVAC_NAME=ppo-dvac-action-adv-fix-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-localshard-v1
CONTROL_RUN="$ROOT/runs/$CONTROL_NAME"
DVAC_RUN="$ROOT/runs/$DVAC_NAME"
CONTROL_PACKET="$ROOT/packets/$CONTROL_NAME"
DVAC_PACKET="$ROOT/packets/$DVAC_NAME"
CONTROL_EXPERIMENT=robotwin_ppo_control_formal100_2gpu64x4_b1024_fixed32_eval5_phys45_localshard_v1
DVAC_EXPERIMENT=robotwin_ppo_dvac_action_adv_fix_w0p5to1p5_formal100_2gpu64x4_b1024_fixed32_eval5_phys67_localshard_v1

test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --short)"
test ! -e "$CONTROL_RUN"
test ! -e "$DVAC_RUN"
test ! -e "$CONTROL_PACKET"
test ! -e "$DVAC_PACKET"
RAY_ADDRESS=172.17.0.1:6389 "$VENV/bin/ray" status >/dev/null

source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment" RLINF_CODE_WORKING_DIR="$WT"
export PYTHONPATH="$WT:$ROBOTWIN" OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

common=(
  --config-path "$WT/examples/embodiment/config"
  --config-name robotwin_adjust_bottle_ppo_openpi_dvac_action_adv
  runner.max_epochs=1000 runner.max_steps=100 runner.val_check_interval=5 runner.save_interval=10 runner.resume_dir=null
  algorithm.update_epoch=2 algorithm.adv_type=gae algorithm.loss_type=actor_critic algorithm.filter_rewards=false
  env.train.total_num_envs=64 env.train.rollout_epoch=4
  env.train.max_episode_steps=200 env.train.max_steps_per_rollout_epoch=200
  "env.train.assets_path=$ROBOTWIN" env.train.video_cfg.save_video=true
  env.eval.total_num_envs=32 env.eval.rollout_epoch=1
  env.eval.max_episode_steps=200 env.eval.max_steps_per_rollout_epoch=200
  env.eval.use_fixed_reset_state_ids=true "env.eval.assets_path=$ROBOTWIN" env.eval.video_cfg.save_video=true
  actor.micro_batch_size=32 actor.global_batch_size=1024
  "actor.model.model_path=$MODEL" actor.fsdp_config.checkpoint_format=local_shard
)

control=(
  'cluster.component_placement={actor\, env\, rollout:"4,5"}'
  "runner.logger.log_path=$CONTROL_RUN" "runner.logger.experiment_name=$CONTROL_EXPERIMENT"
  algorithm.logprob_type=chunk_level
  algorithm.dvac_gradient_weighting.mode=off
  algorithm.dvac_gradient_weighting.application=logprob_st
  algorithm.dvac_gradient_weighting.weight_min=null algorithm.dvac_gradient_weighting.weight_max=null
  "env.train.video_cfg.video_base_dir=$CONTROL_RUN/video/train"
  "env.train.task_config.save_path=$CONTROL_RUN/robotwin_data/train"
  "env.eval.video_cfg.video_base_dir=$CONTROL_RUN/video/eval"
  "env.eval.task_config.save_path=$CONTROL_RUN/robotwin_data/eval"
)

dvac=(
  'cluster.component_placement={actor\, env\, rollout:"6,7"}'
  "runner.logger.log_path=$DVAC_RUN" "runner.logger.experiment_name=$DVAC_EXPERIMENT"
  algorithm.logprob_type=action_level
  algorithm.dvac_gradient_weighting.mode=apply
  algorithm.dvac_gradient_weighting.application=action_advantage
  algorithm.dvac_gradient_weighting.selected_l=3
  algorithm.dvac_gradient_weighting.warmup_steps=1
  algorithm.dvac_gradient_weighting.window_steps=5
  algorithm.dvac_gradient_weighting.weight_min=0.5 algorithm.dvac_gradient_weighting.weight_max=1.5
  "env.train.video_cfg.video_base_dir=$DVAC_RUN/video/train"
  "env.train.task_config.save_path=$DVAC_RUN/robotwin_data/train"
  "env.eval.video_cfg.video_base_dir=$DVAC_RUN/video/eval"
  "env.eval.task_config.save_path=$DVAC_RUN/robotwin_data/eval"
)

mkdir -p "$CONTROL_PACKET" "$DVAC_PACKET"
"$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${common[@]}" "${control[@]}" --cfg job --resolve > "$CONTROL_PACKET/resolved.yaml"
"$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${common[@]}" "${dvac[@]}" --cfg job --resolve > "$DVAC_PACKET/resolved.yaml"

"$VENV/bin/python" - "$CONTROL_PACKET/resolved.yaml" "$DVAC_PACKET/resolved.yaml" "$CONTROL_PACKET/contract.json" "$DVAC_PACKET/contract.json" <<'PY'
import copy, json, sys, yaml
control = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
dvac = yaml.safe_load(open(sys.argv[2], encoding="utf-8"))
for cfg in (control, dvac):
    assert cfg["runner"]["max_steps"] == 100
    assert cfg["runner"]["val_check_interval"] == 5 and cfg["runner"]["save_interval"] == 10
    assert cfg["algorithm"]["adv_type"] == "gae" and cfg["algorithm"]["loss_type"] == "actor_critic"
    assert cfg["algorithm"]["group_size"] == 1 and cfg["algorithm"]["filter_rewards"] is False
    assert cfg["env"]["train"]["total_num_envs"] == 64 and cfg["env"]["train"]["rollout_epoch"] == 4
    assert cfg["env"]["eval"]["total_num_envs"] == 32 and cfg["env"]["eval"]["use_fixed_reset_state_ids"] is True
    assert cfg["actor"]["global_batch_size"] == 1024 and cfg["actor"]["micro_batch_size"] == 32
    assert cfg["actor"]["model"]["add_value_head"] is True
    assert cfg["actor"]["fsdp_config"]["checkpoint_format"] == "local_shard"
assert control["cluster"]["component_placement"] == {"actor, env, rollout": "4,5"}
assert dvac["cluster"]["component_placement"] == {"actor, env, rollout": "6,7"}
assert control["algorithm"]["logprob_type"] == "chunk_level"
assert control["algorithm"]["dvac_gradient_weighting"]["mode"] == "off"
assert dvac["algorithm"]["logprob_type"] == "action_level"
w = dvac["algorithm"]["dvac_gradient_weighting"]
assert (w["mode"], w["application"], w["selected_l"], w["warmup_steps"], w["window_steps"], w["weight_min"], w["weight_max"]) == ("apply", "action_advantage", 3, 1, 5, 0.5, 1.5)

def flatten(obj, prefix=()):
    out = {}
    if isinstance(obj, dict):
        for key, value in obj.items(): out.update(flatten(value, prefix + (str(key),)))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj): out.update(flatten(value, prefix + (str(idx),)))
    else: out[".".join(prefix)] = obj
    return out
a, b = flatten(control), flatten(dvac)
diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
allowed_prefixes = (
    "cluster.component_placement", "runner.logger.log_path", "runner.logger.experiment_name",
    "runner.output_dir", "env.train.video_cfg.video_base_dir", "env.train.task_config.save_path",
    "env.eval.video_cfg.video_base_dir", "env.eval.task_config.save_path",
    "algorithm.logprob_type", "algorithm.dvac_gradient_weighting",
)
unexpected = [k for k in diff if not k.startswith(allowed_prefixes)]
assert not unexpected, unexpected
common_contract = {
    "train_envs": 64, "rollout_epochs": 4, "trajectories_per_step": 256,
    "max_query_records": 1024, "eval_envs": 32, "eval_every": 5,
    "save_every": 10, "global_batch": 1024, "micro_batch": 32, "update_epochs": 2,
    "adv_type": "gae", "loss_type": "actor_critic", "group_size": 1,
    "value_head": True, "checkpoint_format": "local_shard",
}
json.dump({"kind":"PPO Control", "common":common_contract, "allowed_method_diff":diff, "unexpected":unexpected}, open(sys.argv[3],"w"), indent=2)
json.dump({"kind":"PPO DVAC Action-Adv Fix [0.5,1.5]", "common":common_contract, "allowed_method_diff":diff, "unexpected":unexpected}, open(sys.argv[4],"w"), indent=2)
print(json.dumps({"diff_count":len(diff), "unexpected":unexpected, "diff":diff}, indent=2))
PY

printf '%q ' "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${common[@]}" "${control[@]}" > "$CONTROL_PACKET/command.txt"; printf '\n' >> "$CONTROL_PACKET/command.txt"
printf '%q ' "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${common[@]}" "${dvac[@]}" > "$DVAC_PACKET/command.txt"; printf '\n' >> "$DVAC_PACKET/command.txt"
printf '%s\n' "$HEAD" > "$CONTROL_PACKET/source_head.txt"
printf '%s\n' "$HEAD" > "$DVAC_PACKET/source_head.txt"
printf '%s\n' prepared > "$CONTROL_PACKET/packet_complete.txt"
printf '%s\n' prepared > "$DVAC_PACKET/packet_complete.txt"
echo "CONTROL_PACKET=$CONTROL_PACKET"
echo "DVAC_PACKET=$DVAC_PACKET"
echo SZ_PPO_CONTROL_DVAC_MID_FORMAL_PREPARED
