#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421
cd "$WT"

git add -- \
  rlinf/algorithms/rlt/__init__.py \
  rlinf/algorithms/rlt/rollout.py \
  rlinf/algorithms/rlt/route.py \
  rlinf/algorithms/rlt/transition.py \
  rlinf/models/embodiment/openpi/__init__.py \
  rlinf/models/embodiment/openpi/openpi_action_model.py \
  rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py \
  examples/sft/config/robotwin_rlt_stage1_sft_openpi_current_ar.yaml \
  examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_current.yaml \
  tests/unit_tests/test_robotwin_rlt_current_port.py

git diff --cached --check
echo '== diff stat =='
git diff --cached --stat
echo '== name status =='
git diff --cached --name-status
echo '== status =='
git status --short
git diff --cached --binary > /tmp/rlt_current_port.patch
echo "PATCH_BYTES=$(stat -c %s /tmp/rlt_current_port.patch)"
echo 'RLT_STAGED_PATCH_READY'
