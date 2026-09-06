#!/usr/bin/env bash
set -euo pipefail

s05_run=/root/autodl-tmp/experiments/rlt_single_gpu_dvac_pure_reference_bc_s0p5_formal480_20260828_v1
s20_run=/root/autodl-tmp/experiments/rlt_single_gpu_dvac_pure_reference_bc_s2p0_formal480_20260828_v1
s05_runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dvac_pure_reference_bc_s0p5_formal480_20260828_v1/runtime
s20_runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dvac_pure_reference_bc_s2p0_formal480_20260828_v1/runtime
pair_runtime=/root/autodl-tmp/experiment_exports/rlt_dvac_pure_dual_single_gpu_formal480_20260828_v1

for spec in "pure02:$s05_run:$s05_runtime" "pure05:$s20_run:$s20_runtime"; do
  IFS=: read -r label run runtime <<<"$spec"
  echo "=== $label runtime ==="
  find "$runtime" -maxdepth 1 -type f -printf '%s %f\n' | sort -k2
  echo "=== $label selected run files ==="
  find "$run" -type f \( -iname '*trace*.npz' -o -iname '*summary*.json' -o -iname '*config*.yaml' -o -name 'metrics.log' \) -printf '%T@ %s %p\n' | sort -nr | head -30
  echo "=== $label checkpoint dirs ==="
  find "$run" -maxdepth 3 -type d -iname '*checkpoint*' -o -type d -name 'global_step_*' | sort | tail -30
done

echo '=== pair runtime ==='
find "$pair_runtime" -maxdepth 1 -type f -printf '%s %f\n' | sort -k2
