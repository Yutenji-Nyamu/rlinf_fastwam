#!/usr/bin/env bash
set -euo pipefail
chmod +x \
  /tmp/autodl_run_one_rlt_pure_formal480_20260828.sh \
  /tmp/autodl_start_shared_ray_pure_formal480_20260828.sh \
  /tmp/autodl_launch_rlt_pure03_pure04_dual_single_gpu_formal480_20260829.sh
exec /tmp/autodl_launch_rlt_pure03_pure04_dual_single_gpu_formal480_20260829.sh \
  f0aaf4b71669fad38d11ac85c90670386242c29d
