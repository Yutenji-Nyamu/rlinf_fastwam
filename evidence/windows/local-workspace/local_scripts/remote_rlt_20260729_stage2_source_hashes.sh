#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
cd "${repo}"

git status --short --branch
git rev-parse HEAD
sha256sum \
  rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py \
  tests/unit_tests/test_robotwin_rlt_contract.py \
  examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp.yaml \
  examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_a800_2gpu_smoke.yaml
