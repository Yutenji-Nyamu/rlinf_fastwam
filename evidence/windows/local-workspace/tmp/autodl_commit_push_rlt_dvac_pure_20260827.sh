#!/usr/bin/env bash
set -euo pipefail
repo=/root/autodl-tmp/RLinf_rlt_dvac_pure
branch=codex/rlt-dvac-pure-reference-bc
cd "$repo"
test "$(git branch --show-current)" = "$branch"
git diff --check
git add \
  rlinf/algorithms/rlt/dvac_weighting.py \
  rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py \
  tests/unit_tests/test_rlt_dvac_weighting.py \
  examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_dvac_pure_reference_bc_s0p5_mb256_warm20k_replay80k_fresh480.yaml
git diff --cached --check
git commit -m "feat(rlt): reweight successful reference BC with DVAC"
git push personal "HEAD:refs/heads/$branch"
git status --short --branch
git rev-parse HEAD
