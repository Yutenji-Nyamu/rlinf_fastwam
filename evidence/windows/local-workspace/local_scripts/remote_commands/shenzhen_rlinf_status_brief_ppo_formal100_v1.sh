#!/usr/bin/env bash
set -euo pipefail

RUN=/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-formal100-4gpu128train64eval-official-v1
PID="$(cat "$RUN/driver.pid")"

printf 'timestamp=%s driver_pid=%s alive=' "$(date --iso-8601=seconds)" "$PID"
if kill -0 "$PID" 2>/dev/null; then printf 'yes\n'; else printf 'no\n'; fi
nvidia-smi -i 4,5,6,7 \
  --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
free -g | awk 'NR==2 {printf "host_mem_gib total=%s used=%s available=%s\n", $2, $3, $7}'
printf 'fatal_lines=%s\n' "$(grep -Ec 'Traceback|CUDA out of memory|OutOfMemory|SIGKILL|WorkerCrashed|RayActorError' "$RUN/driver.log" || true)"
printf '%s\n' '=== latest stage markers ==='
grep -E 'Creating|initialized|Initialized|rollout|Rollout|generate|Generate|Epoch|Step|Training|Sync|eval/' \
  "$RUN/driver.log" | tail -n 40 || true
printf '%s\n' '=== latest log ==='
tail -n 60 "$RUN/driver.log"
