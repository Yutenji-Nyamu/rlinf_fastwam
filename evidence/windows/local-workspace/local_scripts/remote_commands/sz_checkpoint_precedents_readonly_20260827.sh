#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs
for label_run in \
  "control10:$ROOT/grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2" \
  "old_dvac10:$ROOT/dvac-global-z-w0to2-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v2" \
  "prism_smoke1:$ROOT/prism-dvac-rank-rloo-smoke1-2gpu64x4-b1024-noeval-phys23-v1"; do
  label=${label_run%%:*}; run=${label_run#*:}
  if [[ "$label" == prism_smoke1 ]]; then step=1; else step=10; fi
  dir=$(find "$run" -type d -name "global_step_$step" -print -quit 2>/dev/null || true)
  echo "$label dir=$dir"
  [[ -n "$dir" ]] || continue
  du -sh "$dir"
  find "$dir" -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %f\n' | sort | tail -n 8
done
