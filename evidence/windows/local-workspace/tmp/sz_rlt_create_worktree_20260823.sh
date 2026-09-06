#!/usr/bin/env bash
set -euo pipefail

CANONICAL=/data/chenyiteng/projects/rlinf-shenzhen/RLinf
WORKTREE=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421
BRANCH=codex/sz-rlt-pi0-robotwin-ar
BASE=7d07a4212ee6858cc333e1d4fab7a37256d1f839

test -d "$CANONICAL/.git"
test "$(git -C "$CANONICAL" rev-parse HEAD)" = "$BASE"
test -z "$(git -C "$CANONICAL" status --porcelain=v1)"
git -C "$CANONICAL" cat-file -e "$BASE^{commit}"
test ! -e "$WORKTREE"
if git -C "$CANONICAL" show-ref --verify --quiet "refs/heads/$BRANCH"; then
  echo "TARGET_BRANCH_ALREADY_EXISTS=$BRANCH" >&2
  exit 20
fi

mkdir -p "$(dirname "$WORKTREE")"
git -C "$CANONICAL" worktree add -b "$BRANCH" "$WORKTREE" "$BASE"

test "$(git -C "$WORKTREE" rev-parse HEAD)" = "$BASE"
test -z "$(git -C "$WORKTREE" status --porcelain=v1)"
printf 'RLT_WORKTREE_CREATED\n'
printf 'HEAD=%s\n' "$(git -C "$WORKTREE" rev-parse HEAD)"
printf 'BRANCH=%s\n' "$(git -C "$WORKTREE" branch --show-current)"
printf 'PATH=%s\n' "$WORKTREE"
