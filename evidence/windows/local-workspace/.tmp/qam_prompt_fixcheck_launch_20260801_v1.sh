#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv
run_root=/root/autodl-tmp/experiments/qam_prompt_fixcheck_20260801_v2
experiment=robotwin_adjust_bottle_qam_prompt_fixcheck_20260801_v2
runtime_root=/root/autodl-tmp/experiment_exports/qam_prompt_fixcheck_20260801_v2/runtime
monitor=/root/autodl-tmp/qam_resource_monitor_20260731_v1.sh

test "$(git -C "$repo" branch --show-current)" = codex/qam-pi0-robotwin
test "$(git -C "$repo" rev-parse HEAD)" = d5f6d7d1da0fc355a71ca653be027282cad040d2
test -z "$(git -C "$repo" status --short)"
test ! -e "$run_root"
test ! -e "$runtime_root"
if pgrep -x raylet >/dev/null || pgrep -x gcs_server >/dev/null; then
  printf 'PRECHECK_FAIL=existing_ray\n' >&2
  exit 1
fi
if pgrep -af '[t]rain_embodied_agent.py|[t]orch.distributed.run' >/dev/null; then
  printf 'PRECHECK_FAIL=existing_training\n' >&2
  exit 1
fi

mkdir -p "$runtime_root"
export PYTHONPATH="$repo:/root/autodl-tmp/RoboTwin_RLinf"
export EMBODIED_PATH="$repo/examples/embodiment"
export REPO_PATH="$repo"
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

command=(
  "$venv/bin/python"
  -B
  examples/embodiment/train_embodied_agent.py
  --config-path "$repo/examples/embodiment/config"
  --config-name robotwin_adjust_bottle_qam_openpi
  "runner.logger.log_path=$run_root"
  "runner.logger.experiment_name=$experiment"
  runner.max_steps=4
  runner.save_interval=100
  runner.resume_dir=null
  runner.ckpt_path=null
  env.train.video_cfg.save_video=false
  algorithm.qam.phase=am_on
  algorithm.qam.inv_temp=1.0
  algorithm.qam.warmup_global_inserts=32
  algorithm.qam.q_only_updates_before_am=1
  algorithm.qam.min_replay_per_rank=32
  algorithm.qam.max_updates_per_step=2
  actor.global_batch_size=64
  actor.micro_batch_size=32
  +actor.fsdp_config.save_full_model_weights=false
)

{
  printf 'cd %q\n' "$repo"
  printf 'timeout --signal=TERM --kill-after=180s 1800s'
  printf ' %q' "${command[@]}"
  printf '\n'
} >"$runtime_root/exact_command.txt"

cd "$repo"
set +e
timeout --signal=TERM --kill-after=180s 1800s \
  "${command[@]}" >"$runtime_root/driver.log" 2>&1 &
driver_pid=$!
set -e
printf '%s\n' "$driver_pid" >"$runtime_root/driver.pid"
bash "$monitor" "$driver_pid" "$runtime_root/resources.csv" 10 \
  >"$runtime_root/monitor.log" 2>&1 &
monitor_pid=$!
printf '%s\n' "$monitor_pid" >"$runtime_root/monitor.pid"

set +e
wait "$driver_pid"
status=$?
wait "$monitor_pid"
monitor_status=$?
set -e
printf '%s\n' "$status" >"$runtime_root/exit_code.txt"
printf '%s\n' "$monitor_status" >"$runtime_root/monitor_exit_code.txt"
printf 'QAM_FIXCHECK_EXIT=%s MONITOR_EXIT=%s\n' "$status" "$monitor_status"
exit "$status"
