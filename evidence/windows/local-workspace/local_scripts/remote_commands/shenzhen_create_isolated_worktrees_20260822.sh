#!/usr/bin/env bash
set -euo pipefail

BASE=7d07a4212ee6858cc333e1d4fab7a37256d1f839
RLINF=/data/chenyiteng/projects/rlinf-shenzhen/RLinf
GRPO=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-pi0-robotwin-7d07a421
PI0=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-observe-7d07a421
FASTWAM=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711
FW_BASE=7faa71108368fbb3b6885649f112af607427a2d4
FW_WT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711

printf 'MARKER=SZ_CREATE_ISOLATED_WORKTREES_V1\n'
TZ=Asia/Shanghai date --iso-8601=seconds
id

test "$(git -C "$RLINF" rev-parse HEAD)" = "$BASE"
test -z "$(git -C "$RLINF" status --short)"
test ! -e "$GRPO"
test ! -e "$PI0"
! git -C "$RLINF" show-ref --verify --quiet refs/heads/codex/sz-7d07a421-grpo-pi0-robotwin
! git -C "$RLINF" show-ref --verify --quiet refs/heads/codex/sz-current-pi0-dvac-observe

install -d -m 755 /data/chenyiteng/projects/rlinf-shenzhen/worktrees
git -C "$RLINF" worktree add -b codex/sz-7d07a421-grpo-pi0-robotwin "$GRPO" "$BASE"
git -C "$RLINF" worktree add -b codex/sz-current-pi0-dvac-observe "$PI0" "$BASE"

test "$(git -C "$FASTWAM" rev-parse HEAD)" = "$FW_BASE"
test ! -e "$FW_WT"
! git -C "$FASTWAM" show-ref --verify --quiet refs/heads/codex/sz-fastwam-dvac-observe
install -d -m 755 /data/chenyiteng/projects/fastwam-standalone/worktrees
git -C "$FASTWAM" worktree add -b codex/sz-fastwam-dvac-observe "$FW_WT" "$FW_BASE"

printf '%s\n' '=== RLINF WORKTREES ==='
git -C "$RLINF" worktree list --porcelain
printf '%s\n' '=== FASTWAM WORKTREES ==='
git -C "$FASTWAM" worktree list --porcelain
printf '%s\n' '=== STATUS ==='
git -C "$GRPO" status --short --branch
git -C "$PI0" status --short --branch
git -C "$FW_WT" status --short --branch
printf 'MARKER=SZ_CREATE_ISOLATED_WORKTREES_OK\n'

