#!/usr/bin/env bash
set -euo pipefail

source_root=/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight
config=examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_dvac_r_only_v3_w0to2_100step_formal.yaml

printf 'PRE_COMMIT\n'
git -C "$source_root" status --short
git -C "$source_root" rev-parse HEAD
test "$(git -C "$source_root" rev-parse HEAD)" = 3061872e30cfb496eb296354d30274d35b66576e

git -C "$source_root" add -- "$config"
git -C "$source_root" diff --cached --check
git -C "$source_root" diff --cached --stat
git -C "$source_root" diff --cached -- "$config"
git -C "$source_root" commit -m "config: add DVAC R-only zero-to-two formal run"

printf 'POST_COMMIT\n'
git -C "$source_root" status --short
git -C "$source_root" rev-parse HEAD
