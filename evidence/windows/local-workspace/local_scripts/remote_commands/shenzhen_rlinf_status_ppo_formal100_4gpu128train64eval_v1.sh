#!/usr/bin/env bash
set -euo pipefail

RUN=/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-formal100-4gpu128train64eval-official-v1
PID="$(cat "$RUN/driver.pid")"

printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
printf 'driver_pid=%s\n' "$PID"
if kill -0 "$PID" 2>/dev/null; then
  printf '%s\n' driver_alive=yes
else
  printf '%s\n' driver_alive=no
fi

printf '%s\n' '=== target GPUs 4-7 ==='
nvidia-smi -i 4,5,6,7 \
  --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory \
  --format=csv,noheader,nounits
nvidia-smi -i 4,5,6,7 \
  --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits || true

printf '%s\n' '=== host and Ray ==='
free -h
pgrep -a -x raylet || true
pgrep -a -x gcs_server || true
printf 'run_size=%s\n' "$(du -sh "$RUN" | awk '{print $1}')"

printf '%s\n' '=== progress markers ==='
grep -E 'Creating|Worker|rollout|Rollout|Generate|Training|Epoch|Step|ERROR|Traceback|OutOfMemory|Killed' \
  "$RUN/driver.log" | tail -n 80 || true
printf '%s\n' '=== log tail ==='
tail -n 120 "$RUN/driver.log" || true
