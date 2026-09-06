#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-robotwin-rl
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle@fa8df6ed
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
ROOT=/data/chenyiteng/results/rlinf-shenzhen/pi05
CHAIN="$ROOT/smoke-chain-20260831-v1"
RAY_ADDRESS=172.17.0.1:6389

source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export RAY_ADDRESS ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment" RLINF_CODE_WORKING_DIR="$WT"
export PYTHONPATH="$WT:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

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

run_one() {
  local name=$1 packet_name=$2 config=$3 steps=$4 experiment=$5 kind=$6
  local run="$ROOT/runs/$name" packet="$ROOT/packets/$packet_name"
  test ! -e "$run"
  test -s "$packet/resolved.yaml"
  mkdir -p "$run/runtime"
  cp "$packet/resolved.yaml" "$run/runtime/resolved.yaml"

  local args=(
    --config-path "$WT/examples/embodiment/config" --config-name "$config"
    'cluster.component_placement={actor\, env\, rollout:"2,3"}'
    "runner.logger.log_path=$run" "runner.logger.experiment_name=$experiment"
    runner.max_epochs=1000 runner.max_steps="$steps" runner.val_check_interval=-1 runner.save_interval=1 runner.resume_dir=null
    env.train.total_num_envs=64 env.train.rollout_epoch=4
    env.train.max_episode_steps=200 env.train.max_steps_per_rollout_epoch=200
    "env.train.assets_path=$ROBOTWIN" env.train.video_cfg.save_video=false
    "env.train.task_config.save_path=$run/robotwin_data/train"
    env.eval.total_num_envs=32 env.eval.rollout_epoch=1
    env.eval.max_episode_steps=200 env.eval.max_steps_per_rollout_epoch=200
    env.eval.use_fixed_reset_state_ids=true "env.eval.assets_path=$ROBOTWIN" env.eval.video_cfg.save_video=false
    "env.eval.video_cfg.video_base_dir=$run/video/eval"
    "env.eval.task_config.save_path=$run/robotwin_data/eval"
    actor.micro_batch_size=32 actor.global_batch_size=512
    "actor.model.model_path=$MODEL" actor.model.num_steps=5
    ++actor.fsdp_config.checkpoint_format=local_shard
  )
  if test "$kind" = dvac; then
    args+=(
      algorithm.logprob_type=action_level
      algorithm.dvac_gradient_weighting.mode=apply
      algorithm.dvac_gradient_weighting.application=action_advantage
      algorithm.dvac_gradient_weighting.weight_min=0.5
      algorithm.dvac_gradient_weighting.weight_max=1.5
    )
  fi
  printf '%q ' "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${args[@]}" > "$run/runtime/command.txt"
  printf '\n' >> "$run/runtime/command.txt"
  date --iso-8601=seconds > "$run/runtime/started_at.txt"
  set +e
  timeout --signal=TERM --kill-after=180s 10800s \
    "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${args[@]}" \
    > "$run/runtime/driver.log" 2>&1 &
  local pid=$!
  printf '%s\n' "$pid" > "$run/runtime/driver.pid"
  observe "$pid" "$run/runtime/resource.csv" &
  local observer=$!
  wait "$pid"; local rc=$?
  wait "$observer" || true
  set -e
  printf '%s\n' "$rc" > "$run/runtime/exit_code.txt"
  date --iso-8601=seconds > "$run/runtime/finished_at.txt"
  test "$rc" -eq 0
  local ckpt="$run/checkpoints/global_step_$steps"
  test -s "$ckpt/local_shard_checkpoint/checkpoint_rank_0.pt"
  test -s "$ckpt/local_shard_checkpoint/checkpoint_rank_1.pt"
  if test "$kind" = dvac; then
    test -s "$ckpt/dvac_state_rank0000.json"
    test -s "$ckpt/dvac_state_rank0001.json"
  fi
  printf '%s\n' "$name" >> "$CHAIN/completed.txt"
  sleep 15
}

run_one \
  pi05-ppo-smoke1-2gpu64x4-b512-u5-m5-phys23-localshard-v2 \
  pi05-ppo-smoke1-2gpu64x4-b512-u5-m5-phys23-localshard-v2 \
  robotwin_adjust_bottle_ppo_openpi_pi05 1 pi05_ppo_smoke1 ppo

run_one \
  pi05-grpo-smoke1-2gpu64x4-g8-b512-u5-m5-phys23-localshard-v2 \
  pi05-grpo-smoke1-2gpu64x4-g8-b512-u5-m5-phys23-localshard-v2 \
  robotwin_adjust_bottle_grpo_openpi_pi05 1 pi05_grpo_smoke1 grpo

run_one \
  pi05-grpo-dvac-action-adv-w0p5to1p5-smoke2-2gpu64x4-g8-b512-u5-m5-phys23-localshard-v2 \
  pi05-grpo-dvac-action-adv-w0p5to1p5-smoke2-2gpu64x4-g8-b512-u5-m5-phys23-localshard-v2 \
  robotwin_adjust_bottle_grpo_openpi_pi05 2 pi05_grpo_dvac_action_adv_w0p5to1p5_smoke2 dvac

nvidia-smi -i 4,5,6,7 --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader,nounits > "$CHAIN/gpu4_7_after.txt"
date --iso-8601=seconds > "$CHAIN/finished_at.txt"
printf 'PI05_SMOKE_CHAIN_OK\n'
