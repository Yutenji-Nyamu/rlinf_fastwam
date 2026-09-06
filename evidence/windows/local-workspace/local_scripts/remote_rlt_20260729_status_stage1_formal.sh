#!/usr/bin/env bash
set -u
export LC_ALL=C

RUNTIME_ROOT=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/runtime
RUN_ROOT=/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1
NAME=robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1
DRIVER_LOG=${RUNTIME_ROOT}/driver.log

printf 'OBSERVED_AT\t%s\n' "$(date --iso-8601=seconds)"
driver_pid=$(cat "$RUNTIME_ROOT/driver_pid.txt" 2>/dev/null || printf '%s' -1)
monitor_pid=$(cat "$RUNTIME_ROOT/monitor_pid.txt" 2>/dev/null || printf '%s' -1)
if kill -0 "$driver_pid" 2>/dev/null; then
  printf 'STATE\tRUNNING\n'
else
  printf 'STATE\tFINISHED\n'
fi
printf 'DRIVER_PID\t%s\n' "$driver_pid"
printf 'MONITOR_PID\t%s\n' "$monitor_pid"
if test -f "$RUNTIME_ROOT/exit_code.txt"; then
  printf 'EXIT_CODE\t%s\n' "$(cat "$RUNTIME_ROOT/exit_code.txt")"
fi

printf 'GPU_NOW\n'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
printf 'RAM_NOW\n'
awk '/MemTotal:|MemAvailable:/ {print}' /proc/meminfo
printf 'DISK_NOW\n'
df -hT /root/autodl-tmp
printf 'TRAIN_PROCESSES\n'
pgrep -af 'train_vla_sft|ray::|raylet|gcs_server' || true
printf 'METRIC_LINES\n'
grep -E 'train/(loss|rlt_loss|vla_loss|grad_norm|lr)|Global Step|global_step' \
  "$DRIVER_LOG" 2>/dev/null | tail -n 30 || true
printf 'ERROR_LINES\n'
grep -Ei 'out of memory|CUDA error|Traceback|NaN|Inf|NCCL.*error|ChildFailed|killed' \
  "$DRIVER_LOG" 2>/dev/null | tail -n 30 || true
printf 'LOG_TAIL\n'
tail -n 50 "$DRIVER_LOG" 2>/dev/null || true
printf 'RESOURCE_ROWS\n'
wc -l "$RUNTIME_ROOT/resources.csv" 2>/dev/null || true
tail -n 5 "$RUNTIME_ROOT/resources.csv" 2>/dev/null || true
printf 'CHECKPOINT_DIRS\n'
find "$RUN_ROOT/$NAME/checkpoints" -mindepth 1 -maxdepth 1 -type d \
  -printf '%f\n' 2>/dev/null | sort -V || true
