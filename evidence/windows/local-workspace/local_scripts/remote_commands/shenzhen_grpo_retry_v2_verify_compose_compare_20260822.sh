#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-pi0-robotwin-7d07a421
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
V1=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-formal100-current-4gpu128train64eval-ppo-matched-v1
V2=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-formal100-current-4gpu128train64eval-ppo-matched-v2
PACKET=/data/chenyiteng/results/rlinf-shenzhen/grpo/packets/grpo-formal100-current-4gpu128train64eval-ppo-matched-v2
LAUNCHER="$PACKET/shenzhen_grpo_launch_formal100_retry_v2_20260822.sh"
EXPECTED=5c5cc0b9be56e4c53048a97a6902fe267edeeb63d54b36813294046f6d17d103

test ! -e "$V2"
test -s "$V1/resolved.yaml"
test -f "$LAUNCHER"
test "$(stat -c %s "$LAUNCHER")" = 6061
test "$(sha256sum "$LAUNCHER" | awk '{print $1}')" = "$EXPECTED"
bash -n "$LAUNCHER"
chmod 700 "$LAUNCHER"
printf 'launcher=%s\nlauncher_size=%s\nlauncher_sha256=%s\nlauncher_mode=%s\n' \
  "$LAUNCHER" "$(stat -c %s "$LAUNCHER")" \
  "$(sha256sum "$LAUNCHER" | awk '{print $1}')" "$(stat -c %a "$LAUNCHER")"

source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES
export ROBOTWIN_PATH="$ROBOTWIN"
export ROBOT_PLATFORM=ALOHA
export REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment"
export PYTHONPATH="$WT:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

ARGS=(
  --config-path "$WT/examples/embodiment/config"
  --config-name robotwin_adjust_bottle_grpo_openpi
  'cluster.component_placement={actor\, env\, rollout:4-7}'
  "runner.logger.log_path=$V2"
  'runner.max_epochs=1000'
  'runner.max_steps=100'
  'runner.val_check_interval=10'
  'runner.save_interval=10'
  'runner.resume_dir=null'
  'algorithm.update_epoch=2'
  'env.train.total_num_envs=128'
  'env.train.rollout_epoch=4'
  'env.train.max_episode_steps=200'
  'env.train.max_steps_per_rollout_epoch=200'
  "env.train.assets_path=$ROBOTWIN"
  'env.eval.total_num_envs=64'
  'env.eval.rollout_epoch=1'
  'env.eval.max_episode_steps=200'
  'env.eval.max_steps_per_rollout_epoch=200'
  'env.eval.use_fixed_reset_state_ids=true'
  "env.eval.assets_path=$ROBOTWIN"
  'actor.micro_batch_size=32'
  'actor.global_batch_size=2048'
  "actor.model.model_path=$MODEL"
)

"$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" \
  "${ARGS[@]}" --cfg job --resolve > "$PACKET/v2.prelaunch.resolved.yaml"
sed "s|$V1|$V2|g" "$V1/resolved.yaml" > "$PACKET/v1.normalized-to-v2.resolved.yaml"
if ! cmp -s "$PACKET/v1.normalized-to-v2.resolved.yaml" "$PACKET/v2.prelaunch.resolved.yaml"; then
  printf '%s\n' 'resolved mismatch after normalizing only the run path' >&2
  diff -u "$PACKET/v1.normalized-to-v2.resolved.yaml" "$PACKET/v2.prelaunch.resolved.yaml" | sed -n '1,120p' || true
  exit 1
fi

sha256sum "$V1/resolved.yaml" "$PACKET/v1.normalized-to-v2.resolved.yaml" \
  "$PACKET/v2.prelaunch.resolved.yaml"
printf '%s\n' 'resolved_equal_after_only_v1_to_v2_output_path_substitution=true'
printf '%s\n' 'SZ_GRPO_RETRY_V2_VERIFY_COMPOSE_COMPARE_OK'
