#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs
CONTROL=$ROOT/grpo-control-formal100-2gpu64x4-b1024-eval5-phys45-v1
DVAC=$ROOT/dvac-global-z-w0to2-formal100-2gpu64x4-b1024-eval5-phys67-v1

for tick in $(seq 1 90); do
  cpid=$(<"$CONTROL/runtime/wrapper.pid")
  dpid=$(<"$DVAC/runtime/wrapper.pid")
  kill -0 "$cpid" 2>/dev/null || { echo control_wrapper_exited; tail -n 80 "$CONTROL/runtime/driver.log"; exit 21; }
  kill -0 "$dpid" 2>/dev/null || { echo dvac_wrapper_exited; tail -n 80 "$DVAC/runtime/driver.log"; exit 22; }
  cfatal=$(grep -aEci 'Traceback|OutOfMemory|CUDA out of memory|WorkerCrashed|non[-_ ]?finite|NCCL.*(error|timeout)' "$CONTROL/runtime/driver.log" 2>/dev/null || true)
  dfatal=$(grep -aEci 'Traceback|OutOfMemory|CUDA out of memory|WorkerCrashed|non[-_ ]?finite|NCCL.*(error|timeout)' "$DVAC/runtime/driver.log" 2>/dev/null || true)
  (( cfatal == 0 && dfatal == 0 )) || { echo "fatal control=$cfatal dvac=$dfatal"; exit 23; }
  croll=$(grep -ac 'Generating Rollout Epochs:' "$CONTROL/runtime/driver.log" 2>/dev/null || true)
  droll=$(grep -ac 'Generating Rollout Epochs:' "$DVAC/runtime/driver.log" 2>/dev/null || true)
  if (( croll > 0 && droll > 0 )); then
    echo "both_first_rollout=yes tick=$tick"
    break
  fi
  if (( tick % 3 == 0 )); then
    printf 'waiting tick=%s control_rollout=%s dvac_rollout=%s mem_available_kib=%s\n' "$tick" "$croll" "$droll" "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  fi
  sleep 10
done
test "$croll" -gt 0
test "$droll" -gt 0
echo '=== control rollout tail ==='
grep -aE 'Global Step:|Generating Rollout Epochs:' "$CONTROL/runtime/driver.log" | tail -n 8 || true
echo '=== dvac rollout tail ==='
grep -aE 'Global Step:|Generating Rollout Epochs:' "$DVAC/runtime/driver.log" | tail -n 8 || true
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
free -h | sed -n '1,3p'
echo SZ_DUAL_GRPO_FIRST_ROLLOUT_OK
