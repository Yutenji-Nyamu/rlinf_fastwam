#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current
printf 'MARKER=SZ_APPLY_CURRENT_DVAC_GRPO_STYLE_FIX_V1\n'
git -C "$WT" apply --whitespace=error-all -
git -C "$WT" add -- \
  rlinf/algorithms/dvac_train_weighting.py \
  tests/unit_tests/test_dvac_train_weighting.py
git -C "$WT" diff --cached --check
printf 'MARKER=SZ_APPLY_CURRENT_DVAC_GRPO_STYLE_FIX_OK\n'
