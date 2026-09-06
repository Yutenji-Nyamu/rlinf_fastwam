#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-robotwin-rl
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle@fa8df6ed
HEAD=8c420a9c40f1506521f3146b243dc2021b47a1ae
ROOT=/data/chenyiteng/results/rlinf-shenzhen/pi05
PPO_NAME=pi05-ppo-smoke1-2gpu64x4-b512-u5-m5-phys23-localshard-v2
GRPO_NAME=pi05-grpo-smoke1-2gpu64x4-g8-b512-u5-m5-phys23-localshard-v2
DVAC_NAME=pi05-grpo-dvac-action-adv-w0p5to1p5-smoke2-2gpu64x4-g8-b512-u5-m5-phys23-localshard-v2

test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test "$(git -C "$WT" status --short | wc -l)" -eq 2
test -f "$WT/examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_pi05.yaml"
test ! -e "$WT/examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_pi05_dvac_action_adv.yaml"
RAY_ADDRESS=172.17.0.1:6389 "$VENV/bin/ray" status >/dev/null

source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment" RLINF_CODE_WORKING_DIR="$WT"
export PYTHONPATH="$WT:$ROBOTWIN" OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

common=(
  runner.max_epochs=1000 runner.val_check_interval=-1 runner.save_interval=1 runner.resume_dir=null
  'cluster.component_placement={actor\, env\, rollout:"2,3"}'
  env.train.total_num_envs=64 env.train.rollout_epoch=4
  env.train.max_episode_steps=200 env.train.max_steps_per_rollout_epoch=200
  "env.train.assets_path=$ROBOTWIN" env.train.video_cfg.save_video=false
  env.eval.total_num_envs=32 env.eval.rollout_epoch=1
  env.eval.max_episode_steps=200 env.eval.max_steps_per_rollout_epoch=200
  env.eval.use_fixed_reset_state_ids=true "env.eval.assets_path=$ROBOTWIN" env.eval.video_cfg.save_video=false
  actor.micro_batch_size=32 actor.global_batch_size=512
  "actor.model.model_path=$MODEL" actor.model.num_steps=5
  ++actor.fsdp_config.checkpoint_format=local_shard
)

compose() {
  local config=$1 name=$2 steps=$3 experiment=$4
  shift 4
  local packet="$ROOT/packets/$name" run="$ROOT/runs/$name"
  test ! -e "$packet"; test ! -e "$run"
  mkdir -p "$packet"
  "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" \
    --config-path "$WT/examples/embodiment/config" --config-name "$config" \
    "${common[@]}" runner.max_steps="$steps" \
    "runner.logger.log_path=$run" "runner.logger.experiment_name=$experiment" \
    "env.train.task_config.save_path=$run/robotwin_data/train" \
    "env.eval.task_config.save_path=$run/robotwin_data/eval" \
    "env.eval.video_cfg.video_base_dir=$run/video/eval" \
    "$@" \
    --cfg job --resolve > "$packet/resolved.yaml"
  printf '%s\n' "$config" > "$packet/config_name.txt"
}

compose robotwin_adjust_bottle_ppo_openpi_pi05 "$PPO_NAME" 1 pi05_ppo_smoke1
compose robotwin_adjust_bottle_grpo_openpi_pi05 "$GRPO_NAME" 1 pi05_grpo_smoke1
compose robotwin_adjust_bottle_grpo_openpi_pi05 "$DVAC_NAME" 2 pi05_grpo_dvac_action_adv_w0p5to1p5_smoke2 \
  algorithm.logprob_type=action_level \
  algorithm.dvac_gradient_weighting.mode=apply \
  algorithm.dvac_gradient_weighting.application=action_advantage \
  algorithm.dvac_gradient_weighting.weight_min=0.5 \
  algorithm.dvac_gradient_weighting.weight_max=1.5

"$VENV/bin/python" - \
  "$ROOT/packets/$PPO_NAME/resolved.yaml" \
  "$ROOT/packets/$GRPO_NAME/resolved.yaml" \
  "$ROOT/packets/$DVAC_NAME/resolved.yaml" \
  "$ROOT/packets/pi05-smoke-contract.json" <<'PY'
import json, sys, yaml
ppo, grpo, dvac = [yaml.safe_load(open(p, encoding="utf-8")) for p in sys.argv[1:4]]
for cfg in (ppo, grpo, dvac):
    assert cfg["cluster"]["component_placement"] == {"actor, env, rollout": "2,3"}
    assert cfg["env"]["train"]["total_num_envs"] == 64
    assert cfg["env"]["train"]["rollout_epoch"] == 4
    assert cfg["env"]["eval"]["total_num_envs"] == 32
    assert cfg["actor"]["global_batch_size"] == 512
    assert cfg["actor"]["micro_batch_size"] == 32
    assert cfg["algorithm"]["update_epoch"] == 5
    assert cfg["actor"]["model"]["num_steps"] == 5
    assert cfg["actor"]["optim"]["lr"] == 5e-6
    assert cfg["actor"]["fsdp_config"]["checkpoint_format"] == "local_shard"
    assert cfg["env"]["enable_offload"] is True
    assert cfg["rollout"]["enable_offload"] is True
    assert cfg["actor"]["enable_offload"] is True
assert (ppo["algorithm"]["group_size"], ppo["algorithm"]["adv_type"], ppo["algorithm"]["loss_type"]) == (1, "gae", "actor_critic")
assert ppo["actor"]["model"]["add_value_head"] is True
for cfg in (grpo, dvac):
    assert (cfg["algorithm"]["group_size"], cfg["algorithm"]["adv_type"], cfg["algorithm"]["loss_type"]) == (8, "grpo", "actor")
    assert cfg["algorithm"]["filter_rewards"] is True
    assert cfg["actor"]["model"]["add_value_head"] is False
assert grpo["algorithm"]["logprob_type"] == "chunk_level"
assert grpo["algorithm"]["dvac_gradient_weighting"]["mode"] == "off"
w = dvac["algorithm"]["dvac_gradient_weighting"]
assert dvac["algorithm"]["logprob_type"] == "action_level"
assert (w["mode"], w["application"], w["selected_l"], w["warmup_steps"], w["window_steps"], w["weight_min"], w["weight_max"]) == ("apply", "action_advantage", 3, 1, 5, 0.5, 1.5)
contract = {
    "shared": {"gpus": [2,3], "train_envs": 64, "eval_envs": 32, "rollout_epochs": 4, "trajectories": 256, "max_records": 1024, "global_batch": 512, "micro_batch": 32, "update_epochs": 5, "optimizer_steps_per_outer": 10, "denoise_steps": 5, "checkpoint_format": "local_shard"},
    "ppo_steps": 1, "clean_grpo_steps": 1, "dvac_steps": 2,
}
open(sys.argv[4], "w", encoding="utf-8").write(json.dumps(contract, indent=2) + "\n")
print(json.dumps(contract, indent=2))
PY

printf 'PPO_PACKET=%s\nGRPO_PACKET=%s\nDVAC_PACKET=%s\nPI05_SMOKES_PREPARED\n' \
  "$ROOT/packets/$PPO_NAME" "$ROOT/packets/$GRPO_NAME" "$ROOT/packets/$DVAC_NAME"
