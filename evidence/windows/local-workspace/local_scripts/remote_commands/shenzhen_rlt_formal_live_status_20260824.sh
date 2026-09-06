#!/usr/bin/env bash
set -u

root=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage1-2k-stage2-8env250-20260824-v2
runtime=$root/runtime
stage1_runtime=$root/stage1/runtime

date --iso-8601=seconds
printf '%s\n' '--- owned processes ---'
for item in wrapper resource_observer; do
  file="$runtime/$item.pid"
  if [ -s "$file" ]; then
    pid=$(cat "$file")
    if kill -0 "$pid" 2>/dev/null; then state=alive; else state=dead; fi
    printf '%s pid=%s state=%s\n' "$item" "$pid" "$state"
  fi
done
printf '%s\n' '--- stage markers ---'
for file in "$stage1_runtime/exit_code.txt" "$root/stage2/runtime/started_at.txt" "$runtime/chain_exit.txt"; do
  if [ -s "$file" ]; then printf '%s=' "$file"; tr '\n' ' ' < "$file"; printf '\n'; fi
done
printf '%s\n' '--- recent training lines ---'
grep -E '(^|[^A-Za-z])(step|Step|loss|Loss|grad|tokens|samples|epoch|Epoch)[^A-Za-z]' \
  "$stage1_runtime/driver.log" 2>/dev/null | tail -n 35 || true
printf '%s\n' '--- driver tail ---'
tail -n 35 "$stage1_runtime/driver.log" 2>/dev/null || true
printf '%s\n' '--- GPUs 4-5 ---'
nvidia-smi -i 4,5 --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits
printf '%s\n' '--- host memory ---'
free -h | sed -n '1,2p'
printf '%s\n' '--- resource observer tail ---'
tail -n 8 "$runtime/resource.csv" 2>/dev/null || true
