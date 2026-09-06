#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current
OLD=examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_dvac_global_z.yaml

printf 'MARKER=SZ_FIX_CURRENT_DVAC_GRPO_HYDRA_CONFIG_V1\n'
test -f "$WT/$OLD"
git -C "$WT" rm -f -- "$OLD"
git -C "$WT" apply --index --whitespace=error-all -
git -C "$WT" diff --cached --check
git -C "$WT" status --short
git -C "$WT" diff --cached --stat
printf 'MARKER=SZ_FIX_CURRENT_DVAC_GRPO_HYDRA_CONFIG_OK\n'
