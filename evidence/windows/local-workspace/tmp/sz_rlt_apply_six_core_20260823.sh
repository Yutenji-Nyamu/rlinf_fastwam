#!/usr/bin/env bash
set -euo pipefail

WORKTREE=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421
BASE=7d07a4212ee6858cc333e1d4fab7a37256d1f839

test "$(git -C "$WORKTREE" rev-parse HEAD)" = "$BASE"
test -z "$(git -C "$WORKTREE" status --porcelain=v1)"

git -C "$WORKTREE" apply --3way -
git -C "$WORKTREE" diff --check
git -C "$WORKTREE" status --short
