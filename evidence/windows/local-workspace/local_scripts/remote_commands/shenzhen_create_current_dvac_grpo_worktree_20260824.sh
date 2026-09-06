#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/projects/rlinf-shenzhen/RLinf
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current
BRANCH=codex/sz-current-pi0-dvac-grpo
BASE=554c6dc8d586162d9444c01fa88308ed4f5203d0
TELEMETRY=f7cf0f6092b92e0ab2b813bc8acd8ee132a288e5

printf 'MARKER=SZ_CREATE_CURRENT_DVAC_GRPO_V1\n'
TZ=Asia/Shanghai date --iso-8601=seconds
id
git -C "$ROOT" cat-file -e "$BASE^{commit}"
git -C "$ROOT" cat-file -e "$TELEMETRY^{commit}"
test ! -e "$WT"
! git -C "$ROOT" show-ref --verify --quiet "refs/heads/$BRANCH"
git -C "$ROOT" worktree add -b "$BRANCH" "$WT" "$BASE"
git -C "$WT" cherry-pick "$TELEMETRY"
test -z "$(git -C "$WT" status --short)"
printf 'worktree=%s\nbranch=%s\nhead=%s\n' \
  "$WT" "$(git -C "$WT" branch --show-current)" "$(git -C "$WT" rev-parse HEAD)"
git -C "$WT" log --oneline --decorate -3
printf 'MARKER=SZ_CREATE_CURRENT_DVAC_GRPO_OK\n'
