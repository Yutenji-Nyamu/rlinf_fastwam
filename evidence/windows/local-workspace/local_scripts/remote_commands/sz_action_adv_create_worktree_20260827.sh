#!/usr/bin/env bash
set -euo pipefail

SOURCE=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current
TARGET=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-dvac-action-adv
BASE=0e28ac6f09f821ea12e7d54eba7118ce0000ca86
BRANCH=codex/sz-grpo-dvac-action-adv

test "$(git -C "$SOURCE" rev-parse HEAD)" = "$BASE"
test -z "$(git -C "$SOURCE" status --short)"
test ! -e "$TARGET"
! git -C "$SOURCE" show-ref --verify --quiet "refs/heads/$BRANCH"
test -z "$(git -C "$SOURCE" ls-remote --heads personal "$BRANCH")"

git -C "$SOURCE" worktree add -b "$BRANCH" "$TARGET" "$BASE"
test "$(git -C "$TARGET" rev-parse HEAD)" = "$BASE"
test "$(git -C "$TARGET" branch --show-current)" = "$BRANCH"
test -z "$(git -C "$TARGET" status --short)"
git -C "$TARGET" log -1 --oneline
git -C "$TARGET" status --short
echo SZ_ACTION_ADV_WORKTREE_CREATED
