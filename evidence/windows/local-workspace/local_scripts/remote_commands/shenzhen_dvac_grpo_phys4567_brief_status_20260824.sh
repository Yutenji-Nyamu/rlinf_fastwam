#!/usr/bin/env bash
set -u

RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-4gpu32x8-g8-phys4567-v3
LOG="$RUN/runtime/driver.log"

echo '=== identity/time ==='
date '+%F %T %Z'
id -un

echo '=== process ==='
if [[ -f "$RUN/runtime/wrapper.pid" ]]; then
  pid=$(<"$RUN/runtime/wrapper.pid")
  if kill -0 "$pid" 2>/dev/null; then echo "wrapper_alive=yes pid=$pid"; else echo "wrapper_alive=no pid=$pid"; fi
else
  echo 'wrapper_pid=missing'
fi
if [[ -f "$RUN/runtime/exit_code.txt" ]]; then printf 'exit_code='; cat "$RUN/runtime/exit_code.txt"; else echo 'exit_code=pending'; fi

echo '=== latest metrics/progress ==='
if [[ -f "$LOG" ]]; then
  grep -E 'Global Step:|Elapsed:|Step Time:|success_once=|train/(kl_loss|clip_fraction|grad_norm)|dvac/(weight_mean|weight_std|weight_ess_fraction|z_clip_low_fraction|z_clip_high_fraction)' "$LOG" | tail -n 80 || true
  grep -E 'Generating Rollout Epochs:' "$LOG" | tail -n 4 || true
  printf 'fatal_matches='
  grep -Eci 'Traceback|OutOfMemory|CUDA out of memory|WorkerCrashed|non[-_ ]?finite|NCCL.*(error|timeout)' "$LOG" || true
else
  echo 'driver_log=missing'
fi

echo '=== checkpoints ==='
find "$RUN" -type d -name 'global_step_*' -printf '%f\n' 2>/dev/null | sort -V | tail -n 12

echo '=== resources ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
free -h | sed -n '1,2p'
if [[ -f "$RUN/runtime/resource.csv" ]]; then tail -n 2 "$RUN/runtime/resource.csv"; fi
