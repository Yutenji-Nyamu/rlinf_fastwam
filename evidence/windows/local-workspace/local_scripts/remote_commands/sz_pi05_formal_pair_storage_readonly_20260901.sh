#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/chenyiteng/results/rlinf-shenzhen/pi05/runs
for run in \
  "$ROOT/pi05-grpo-smoke1-2gpu64x4-g8-b512-u5-m5-phys23-localshard-v2" \
  "$ROOT/pi05-grpo-dvac-action-adv-w0p5to1p5-smoke2-2gpu64x4-g8-b512-u5-m5-phys23-localshard-v2" \
  /data/chenyiteng/results/rlinf-shenzhen/ppo/runs/ppo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-localshard-v1; do
  printf '%s\n' "=== $run ==="
  du -sh "$run"
  find "$run" -type d -name 'global_step_*' -print0 | xargs -0 -r du -sh | sort -V | tail -n 12
done
df -h /data
