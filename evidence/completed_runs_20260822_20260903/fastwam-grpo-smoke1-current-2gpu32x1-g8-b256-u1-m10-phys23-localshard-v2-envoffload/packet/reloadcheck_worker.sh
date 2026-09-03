#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
FW=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711/src
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
ROOT=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo
NAME=fastwam-grpo-smoke1-current-2gpu32x1-g8-b256-u1-m10-phys23-localshard-v2-envoffload
RUN="$ROOT/runs/$NAME"
PACKET="$ROOT/packets/$NAME"
RELOAD="$ROOT/runs/${NAME}-reloadcheck"
CKPT="$RUN/fastwam_grpo_smoke1_current/checkpoints/global_step_1"
RAY_ADDRESS=172.17.0.1:6389

test -s "$CKPT/actor/local_shard_checkpoint/checkpoint_rank_0.pt"
test -s "$CKPT/actor/local_shard_checkpoint/checkpoint_rank_1.pt"
test ! -e "$RELOAD"
source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export RAY_ADDRESS ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment" RLINF_CODE_WORKING_DIR="$WT"
export PYTHONPATH="$WT:$FW:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export DIFFSYNTH_MODEL_BASE_PATH=/data/chenyiteng/models/fastwam/diffsynth
export MODELSCOPE_CACHE=/home/chenyiteng/cache/fastwam-7faa/modelscope
export TMPDIR=/home/chenyiteng/cache/fastwam-7faa/tmp
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

mkdir -p "$RELOAD/runtime"
date --iso-8601=seconds > "$RELOAD/runtime/started_at.txt"
set +e
timeout --signal=TERM --kill-after=180s 1800s \
  "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" \
  --config-path "$WT/examples/embodiment/config" \
  --config-name robotwin_move_stapler_pad_grpo_fastwam \
  'cluster.component_placement={actor\, env\, rollout:"2,3"}' \
  runner.max_epochs=1000 runner.max_steps=1 runner.val_check_interval=1 runner.save_interval=1 \
  "runner.resume_dir=$CKPT" "runner.logger.log_path=$RELOAD" \
  runner.logger.experiment_name=fastwam_grpo_smoke1_reloadcheck \
  env.train.total_num_envs=32 env.train.rollout_epoch=1 \
  env.train.max_episode_steps=192 env.train.max_steps_per_rollout_epoch=192 env.train.video_cfg.save_video=false \
  "env.train.assets_path=$ROBOTWIN" "env.train.task_config.save_path=$RELOAD/robotwin_data/train" \
  env.eval.total_num_envs=32 env.eval.rollout_epoch=1 \
  env.eval.max_episode_steps=192 env.eval.max_steps_per_rollout_epoch=192 \
  env.eval.use_fixed_reset_state_ids=true env.eval.video_cfg.save_video=false \
  "env.eval.assets_path=$ROBOTWIN" "env.eval.task_config.save_path=$RELOAD/robotwin_data/eval" \
  "env.eval.video_cfg.video_base_dir=$RELOAD/video/eval" \
  actor.micro_batch_size=2 actor.global_batch_size=256 \
  algorithm.group_size=8 algorithm.update_epoch=1 \
  actor.fsdp_config.checkpoint_format=local_shard actor.fsdp_config.cast_forward_inputs=false \
  > "$RELOAD/runtime/driver.log" 2>&1
rc=$?
set -e
printf '%s\n' "$rc" > "$RELOAD/runtime/exit_code.txt"
date --iso-8601=seconds > "$RELOAD/runtime/finished_at.txt"
test "$rc" -eq 0
grep -F "Resuming training from checkpoint directory $CKPT" "$RELOAD/runtime/driver.log" >/dev/null
grep -F 'Training finished!' "$RELOAD/runtime/driver.log" >/dev/null
date --iso-8601=seconds > "$PACKET/SMOKE_OK"
printf 'FASTWAM_CURRENT_GRPO_RELOAD_OK\n'
