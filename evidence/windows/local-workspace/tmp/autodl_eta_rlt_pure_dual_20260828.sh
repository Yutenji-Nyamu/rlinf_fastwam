#!/usr/bin/env bash
set -u

for spec in \
  "s05:/root/autodl-tmp/experiment_exports/rlt_single_gpu_dvac_pure_reference_bc_s0p5_formal480_20260828_v1/runtime" \
  "s20:/root/autodl-tmp/experiment_exports/rlt_single_gpu_dvac_pure_reference_bc_s2p0_formal480_20260828_v1/runtime"; do
  label=${spec%%:*}
  runtime=${spec#*:}
  echo "=== $label ==="
  grep -a -E 'Elapsed:|Global Step:' "$runtime/foreground.log" 2>/dev/null | tail -n 12 || true
done
