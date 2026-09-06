#!/usr/bin/env bash
set -euo pipefail

base=/data/chenyiteng/results/rlinf-shenzhen/fastwam-action-dvac-adv/runs/fastwam-action-dvac-adv-w05to15-smoke2-2gpu32x1-g8-b256-u1-m10-phys23-dcp-v2

printf '%s\n' 'WEIGHT_SUMMARY'
cat "$base-reload-step2/runtime/weight_summary.json"
printf '%s\n' 'MARKERS'
find "$base" "$base-reload-step2" -maxdepth 3 -type f \( -name 'SMOKE_OK' -o -name 'exit_code.txt' \) -print -exec cat {} \;
printf '%s\n' 'GPU_SNAPSHOT'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
printf '%s\n' 'FORMAL_GPU_4_TO_7_PROCESSES'
ps -eo pid=,etimes=,args= | grep -E 'ppo|PPO' | grep -v grep | head -n 30 || true
printf '%s\n' 'FATAL_COUNTS'
for log in "$base"/driver.log "$base-reload-step2"/driver.log; do
  printf '%s=' "$log"
  grep -Eic 'Traceback|CUDA out of memory|OutOfMemoryError|NCCL.*(error|failed)|nonfinite|NaN|Fatal Python error' "$log" || true
done
