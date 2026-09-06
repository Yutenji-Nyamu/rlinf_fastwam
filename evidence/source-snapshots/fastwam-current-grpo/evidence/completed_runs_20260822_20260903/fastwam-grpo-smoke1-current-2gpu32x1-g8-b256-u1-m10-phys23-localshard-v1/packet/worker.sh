#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
FW=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711/src
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
ROOT=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo
NAME=fastwam-grpo-smoke1-current-2gpu32x1-g8-b256-u1-m10-phys23-localshard-v1
RUN="$ROOT/runs/$NAME"
PACKET="$ROOT/packets/$NAME"
RELOAD="$ROOT/runs/${NAME}-reloadcheck"
RAY_ADDRESS=172.17.0.1:6389

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

observe() {
  local pid=$1 out=$2
  printf '%s\n' 'timestamp,driver_alive,host_mem_available_kib,gpu2_used_mib,gpu2_util_pct,gpu3_used_mib,gpu3_util_pct' > "$out"
  while kill -0 "$pid" 2>/dev/null; do
    printf '%s,1,%s' "$(date --iso-8601=seconds)" "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" >> "$out"
    while IFS= read -r row; do printf ',%s' "$(tr -d ' ' <<< "$row")" >> "$out"; done < <(nvidia-smi -i 2,3 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    printf '\n' >> "$out"
    sleep 30
  done
  printf '%s,0,%s\n' "$(date --iso-8601=seconds)" "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" >> "$out"
}

common=(
  --config-path "$WT/examples/embodiment/config"
  --config-name robotwin_move_stapler_pad_grpo_fastwam
  'cluster.component_placement={actor\, env\, rollout:"2,3"}'
  runner.max_epochs=1000 runner.max_steps=1 runner.val_check_interval=1 runner.save_interval=1
  env.train.total_num_envs=32 env.train.rollout_epoch=1
  env.train.max_episode_steps=192 env.train.max_steps_per_rollout_epoch=192 env.train.video_cfg.save_video=false
  "env.train.assets_path=$ROBOTWIN"
  env.eval.total_num_envs=32 env.eval.rollout_epoch=1
  env.eval.max_episode_steps=192 env.eval.max_steps_per_rollout_epoch=192
  env.eval.use_fixed_reset_state_ids=true env.eval.video_cfg.save_video=false
  "env.eval.assets_path=$ROBOTWIN"
  actor.micro_batch_size=2 actor.global_batch_size=256
  algorithm.group_size=8 algorithm.update_epoch=1
  actor.fsdp_config.checkpoint_format=local_shard actor.fsdp_config.cast_forward_inputs=false
)

mkdir -p "$RUN/runtime"
cp "$PACKET/resolved.yaml" "$RUN/runtime/resolved.yaml"
cp "$PACKET/contract.json" "$RUN/runtime/contract.json"
cp "$PACKET/command.txt" "$RUN/runtime/command.txt"
date --iso-8601=seconds > "$RUN/runtime/started_at.txt"
set +e
timeout --signal=TERM --kill-after=180s 21600s \
  "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" \
  "${common[@]}" runner.resume_dir=null \
  "runner.logger.log_path=$RUN" runner.logger.experiment_name=fastwam_grpo_smoke1_current \
  "env.train.task_config.save_path=$RUN/robotwin_data/train" \
  "env.eval.task_config.save_path=$RUN/robotwin_data/eval" \
  "env.eval.video_cfg.video_base_dir=$RUN/video/eval" \
  > "$RUN/runtime/driver.log" 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$RUN/runtime/driver.pid"
observe "$pid" "$RUN/runtime/resource.csv" & observer=$!
wait "$pid"; rc=$?
wait "$observer" || true
set -e
printf '%s\n' "$rc" > "$RUN/runtime/exit_code.txt"
date --iso-8601=seconds > "$RUN/runtime/finished_at.txt"
test "$rc" -eq 0
CKPT="$RUN/checkpoints/global_step_1"
test -s "$CKPT/actor/local_shard_checkpoint/checkpoint_rank_0.pt"
test -s "$CKPT/actor/local_shard_checkpoint/checkpoint_rank_1.pt"

sleep 20
mkdir -p "$RELOAD/runtime"
date --iso-8601=seconds > "$RELOAD/runtime/started_at.txt"
set +e
timeout --signal=TERM --kill-after=180s 1800s \
  "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" \
  "${common[@]}" "runner.resume_dir=$CKPT" \
  "runner.logger.log_path=$RELOAD" runner.logger.experiment_name=fastwam_grpo_smoke1_reloadcheck \
  "env.train.task_config.save_path=$RELOAD/robotwin_data/train" \
  "env.eval.task_config.save_path=$RELOAD/robotwin_data/eval" \
  "env.eval.video_cfg.video_base_dir=$RELOAD/video/eval" \
  > "$RELOAD/runtime/driver.log" 2>&1 &
reload_pid=$!
printf '%s\n' "$reload_pid" > "$RELOAD/runtime/driver.pid"
wait "$reload_pid"; reload_rc=$?
set -e
printf '%s\n' "$reload_rc" > "$RELOAD/runtime/exit_code.txt"
date --iso-8601=seconds > "$RELOAD/runtime/finished_at.txt"
test "$reload_rc" -eq 0
grep -F "Resuming training from checkpoint directory $CKPT" "$RELOAD/runtime/driver.log" >/dev/null
grep -F 'Training finished!' "$RELOAD/runtime/driver.log" >/dev/null
date --iso-8601=seconds > "$PACKET/SMOKE_OK"
printf 'FASTWAM_CURRENT_GRPO_SMOKE_OK\n'
