#!/usr/bin/env bash
set -euo pipefail
for run in \
  /data/chenyiteng/results/rlinf-rlt/smokes/current-single-gpu-control-phys2-1cycle-20260830-v2 \
  /data/chenyiteng/results/rlinf-rlt/smokes/current-single-gpu-pure04-phys3-1cycle-20260830-v2; do
  printf '\n%s\n' "$run"
  find "$run" -path '*global_step_1*' -type f -printf '%P\n' | sort | head -n 80
done
