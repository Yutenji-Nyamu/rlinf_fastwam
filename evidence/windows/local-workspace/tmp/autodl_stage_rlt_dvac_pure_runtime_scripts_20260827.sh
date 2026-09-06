#!/usr/bin/env bash
set -euo pipefail
tar -xf /tmp/rlt_dvac_pure_runtime_scripts_20260827.tar -C /tmp
for script in \
  /tmp/autodl_rlt_dvac_pure_pretest_20260827.sh \
  /tmp/autodl_run_rlt_dvac_pure_smoke_20260827.sh \
  /tmp/autodl_collect_rlt_matched_width_final_20260827.sh; do
  bash -n "$script"
done
echo RLT_DVAC_PURE_RUNTIME_SCRIPTS_STAGED
