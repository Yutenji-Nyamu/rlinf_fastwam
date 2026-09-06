#!/usr/bin/env bash
set -euo pipefail
for run in \
 /data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2 \
 /data/chenyiteng/results/rlinf-shenzhen/pi05/runs/pi05-grpo-control-formal100-2gpu64x4-g8-b1024-u2-m5-fixed32-eval5-phys45-localshard-v2 \
 /data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-formal100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-v1 \
 /data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage2-8env250-20260824-v4-warmup-fix \
 /data/chenyiteng/results/rlinf-current-dsrl/formal-dsrl-pi0-robotwin-2gpu4env-200c-20260824-v2; do
  printf '\nRUN=%s\n' "$run"
  test -d "$run" || { echo MISSING; continue; }
  find "$run" -path '*/checkpoints/*' -prune -o -path '*/video/*' -prune -o -path '*/videos/*' -prune -o -path '*/robotwin_data/*' -prune -o -type f -printf '%s\t%P\n' | sort -nr | sed -n '1,45p'
done
