#!/usr/bin/env bash
set -euo pipefail
repo=/root/autodl-tmp/RLinf_rlt_dvac_success_bc
echo IDENTITY
date -Is
hostname
id -u
echo GIT
git -C "$repo" status --short --branch
git -C "$repo" rev-parse HEAD
git -C "$repo" log -4 --oneline --decorate
echo HASHES
sha256sum \
  "$repo/rlinf/algorithms/rlt/dvac_weighting.py" \
  "$repo/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py" \
  "$repo/tests/unit_tests/test_rlt_dvac_weighting.py" \
  "$repo/examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_fresh480_mb256_warm20k_replay80k_control.yaml"
