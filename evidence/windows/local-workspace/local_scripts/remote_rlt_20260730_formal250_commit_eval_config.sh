#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
cd "$repo"

test "$(git branch --show-current)" = codex/rlt-pi0-robotwin
test "$(git rev-parse HEAD)" = \
  2df23e7f4b3d19d4f0dedab32168767a32845a58

expected_status=' M tests/unit_tests/test_robotwin_seed_partition.py
?? examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250.yaml
?? examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_resource_smoke.yaml
?? rlinf/envs/robotwin/seeds/eval_seeds_adjust_bottle_rlt_periodic20_v1.json'
test "$(git status --short)" = "$expected_status"

test "$(sha256sum rlinf/envs/robotwin/seeds/eval_seeds_adjust_bottle_rlt_periodic20_v1.json | awk '{print $1}')" = \
  fb9c3353e27b83aad6fe7ff778437d960b084de9d981c2af68615d52769952a7
test "$(sha256sum examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250.yaml | awk '{print $1}')" = \
  d9ee30f8c776b349cc3e5e08cb17c97f46151c9c62bffada7f514e9522ffb315
test "$(sha256sum examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_resource_smoke.yaml | awk '{print $1}')" = \
  47bbdd231d53d69aad5aee35ef6c04787d6d9c7610f02df9f945961466d6acb2
test "$(sha256sum tests/unit_tests/test_robotwin_seed_partition.py | awk '{print $1}')" = \
  4d764fc0a0e042e1b71a93d98d5e9a2627e6ed8a899c9dee8a92528303ad3c0f

git diff --check
git add -- \
  examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250.yaml \
  examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_resource_smoke.yaml \
  rlinf/envs/robotwin/seeds/eval_seeds_adjust_bottle_rlt_periodic20_v1.json \
  tests/unit_tests/test_robotwin_seed_partition.py
git diff --cached --check
git commit -m "feat(rlt): add 8-env formal evaluation protocol"

test -z "$(git status --short)"
git show --stat --oneline --decorate --no-renames HEAD
