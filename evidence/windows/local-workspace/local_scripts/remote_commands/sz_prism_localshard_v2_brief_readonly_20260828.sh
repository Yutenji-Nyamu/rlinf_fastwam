#!/usr/bin/env bash
set -euo pipefail
RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/prism-dvac-rank-rloo-formal100-2gpu64x4-b1024-fixed32-eval5-localshard-phys67-v2
CONTROL=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2
for item in prism:$RUN control:$CONTROL; do
  name=${item%%:*}; path=${item#*:}; pid=$(cat "$path/runtime/wrapper.pid")
  kill -0 "$pid" 2>/dev/null && alive=yes || alive=no
  step=$(grep -aoE 'Global Step:[[:space:]]+[0-9]+/100' "$path/runtime/driver.log" 2>/dev/null | tail -n1 || true)
  printf '%s pid=%s alive=%s latest=%s\n' "$name" "$pid" "$alive" "${step:-none}"
done
grep -E 'Starting rollout|Rollout epoch|Begin rollout|Global Step:|Finished rollout|optimizer' "$RUN/runtime/driver.log" 2>/dev/null | tail -n 20 || true
printf 'fatal_count='
grep -Eic 'Traceback|out of memory|CUDA error|RayActorError|worker died|ErrorInitializationFailed|NCCL.*error' "$RUN/runtime/driver.log" 2>/dev/null || true
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
awk '/^MemAvailable:/ {print}' /proc/meminfo
