#!/usr/bin/env bash
set -euo pipefail
RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-w0to5-formal100-grpo-matched-4gpu128x4-b2048-eval5-phys4567-v1
TZ=Asia/Shanghai date --iso-8601=seconds
cat "$RUN/runtime/stopped_by_user_for_dual_2gpu.txt"
echo '=== runtime files ==='
find "$RUN/runtime" -maxdepth 1 -type f -printf '%s %p\n' | sort -n
echo '=== tensorboard candidates ==='
find "$RUN" -type f \( -name 'events.out.tfevents.*' -o -path '*/tensorboard/config.yaml' \) -printf '%s %p\n' | sort -n
echo '=== total ==='
du -sh "$RUN"
