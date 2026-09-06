#!/usr/bin/env bash
set -euo pipefail

RUN=${RUN:-/data/chenyiteng/results/rlinf-shenzhen/ppo/sft-fixed64-4gpu4567-official-half-v1}
printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
printf '%s\n' '=== gpu 0-7 ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory \
  --format=csv,noheader,nounits
printf '%s\n' '=== compute apps ==='
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits || true
printf '%s\n' '=== host memory ==='
free -h
printf '%s\n' '=== ray controls ==='
pgrep -a -x raylet || true
pgrep -a -x gcs_server || true
printf '%s\n' '=== run ==='
du -sh "$RUN" 2>/dev/null || true
tail -n 80 "$RUN/driver.log" 2>/dev/null || true
