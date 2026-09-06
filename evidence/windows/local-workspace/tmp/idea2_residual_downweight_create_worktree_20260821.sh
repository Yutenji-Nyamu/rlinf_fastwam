#!/usr/bin/env bash
set -euo pipefail

SOURCE=/root/autodl-tmp/RLinf_idea2_dvac_train
TARGET=/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight
BRANCH=codex/idea2-dvac-residual-downweight
BASE=145fa810f1d8baee23012922b81e496661d61cf5

test ! -e "$TARGET"
git -C "$SOURCE" worktree add -b "$BRANCH" "$TARGET" "$BASE"
git -C "$TARGET" rev-parse HEAD
git -C "$TARGET" branch --show-current
git -C "$TARGET" status --short --branch
