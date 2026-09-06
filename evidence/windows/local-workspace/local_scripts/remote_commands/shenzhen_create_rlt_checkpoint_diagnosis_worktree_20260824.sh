#!/usr/bin/env bash
set -euo pipefail

source_worktree=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421
target_worktree=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-checkpoint-diagnosis-7d07a421
branch=codex/sz-rlt-checkpoint-diagnosis
expected_head=f3ea5f691b99fe39e024e5571c0e6ee3d83c51b4

test "$(git -C "$source_worktree" rev-parse HEAD)" = "$expected_head"
test -z "$(git -C "$source_worktree" status --porcelain)"
test ! -e "$target_worktree"
if git -C "$source_worktree" show-ref --verify --quiet "refs/heads/$branch"; then
  printf '%s\n' "branch already exists: $branch" >&2
  exit 2
fi

git -C "$source_worktree" worktree add -b "$branch" "$target_worktree" "$expected_head"
test "$(git -C "$target_worktree" rev-parse HEAD)" = "$expected_head"
test "$(git -C "$target_worktree" branch --show-current)" = "$branch"
test -z "$(git -C "$target_worktree" status --porcelain)"

printf 'worktree=%s\nbranch=%s\nhead=%s\n' \
  "$target_worktree" "$branch" "$expected_head"
printf '%s\n' 'SZ_RLT_CHECKPOINT_DIAGNOSIS_WORKTREE_CREATED'
