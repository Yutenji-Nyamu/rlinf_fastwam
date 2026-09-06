#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_rlt_teacher_dvac
expected_branch=codex/rlt-teacher-dvac-weighting
expected_base=2b8199d8ab2e7b110994fd3234bf7007196c3af9

test "$(git -C "$repo" branch --show-current)" = "$expected_branch"
test "$(git -C "$repo" rev-parse HEAD)" = "$expected_base"
git -C "$repo" diff --check

git -C "$repo" add -- \
  rlinf/algorithms/rlt/dvac_weighting.py \
  rlinf/algorithms/rlt/rollout.py \
  rlinf/algorithms/rlt/transition.py \
  rlinf/models/embodiment/openpi/openpi_action_model.py \
  rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py \
  examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250_teacher_dvac_w0to2.yaml \
  tests/unit_tests/test_rlt_dvac_weighting.py

printf 'STAGED_FILES\n'
git -C "$repo" diff --cached --name-status
test "$(git -C "$repo" diff --cached --name-only | wc -l)" -eq 7

git -C "$repo" commit -m 'feat(rlt): add teacher DVAC Q-gradient weighting'
printf 'COMMIT=%s\n' "$(git -C "$repo" rev-parse HEAD)"
git -C "$repo" status --short --branch
