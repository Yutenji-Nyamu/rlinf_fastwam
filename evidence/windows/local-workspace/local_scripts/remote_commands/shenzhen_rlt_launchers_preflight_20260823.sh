#!/usr/bin/env bash
set -euo pipefail

launcher_dir=/data/chenyiteng/results/rlinf-rlt/launchers

chmod 700 \
  "$launcher_dir/shenzhen_rlt_current_smoke_observer_gpu4_5_20260823.sh" \
  "$launcher_dir/shenzhen_rlt_current_stage1_smoke2_gpu4_5_20260823.sh" \
  "$launcher_dir/shenzhen_rlt_current_stage2_smoke_fresh_gpu4_5_20260823.sh" \
  "$launcher_dir/shenzhen_rlt_current_stage2_smoke_resume_gpu4_5_20260823.sh"

bash -n "$launcher_dir/shenzhen_rlt_current_smoke_observer_gpu4_5_20260823.sh"
bash -n "$launcher_dir/shenzhen_rlt_current_stage1_smoke2_gpu4_5_20260823.sh"
bash -n "$launcher_dir/shenzhen_rlt_current_stage2_smoke_fresh_gpu4_5_20260823.sh"
bash -n "$launcher_dir/shenzhen_rlt_current_stage2_smoke_resume_gpu4_5_20260823.sh"

test -s /data/chenyiteng/datasets/robotwin2/canonical/pi0-aloha-clean50-v1/rlt_canonical_validation.json

sha256sum \
  "$launcher_dir/shenzhen_rlt_current_smoke_observer_gpu4_5_20260823.sh" \
  "$launcher_dir/shenzhen_rlt_current_stage1_smoke2_gpu4_5_20260823.sh" \
  "$launcher_dir/shenzhen_rlt_current_stage2_smoke_fresh_gpu4_5_20260823.sh" \
  "$launcher_dir/shenzhen_rlt_current_stage2_smoke_resume_gpu4_5_20260823.sh"

printf '%s\n' 'RLT_LAUNCHERS_PREFLIGHT_OK'
