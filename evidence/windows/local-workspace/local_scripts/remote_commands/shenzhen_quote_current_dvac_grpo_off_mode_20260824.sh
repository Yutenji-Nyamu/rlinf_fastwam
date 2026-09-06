#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current
CFG=examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi.yaml
printf 'MARKER=SZ_QUOTE_CURRENT_DVAC_GRPO_OFF_MODE_V1\n'
git -C "$WT" apply --whitespace=error-all -
git -C "$WT" add -- "$CFG"
git -C "$WT" diff --cached --check
grep -A2 'dvac_gradient_weighting:' "$WT/$CFG"
printf 'MARKER=SZ_QUOTE_CURRENT_DVAC_GRPO_OFF_MODE_OK\n'
