#!/usr/bin/env bash
set -euo pipefail

repo=/data/chenyiteng/projects/rlinf-shenzhen/RLinf
target=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/prism-dvac-rank-rloo
branch=codex/sz-prism-dvac-rank-rloo
base=0e28ac6f09f821ea12e7d54eba7118ce0000ca86

test "$(git -C "$repo" cat-file -t "$base")" = commit
test ! -e "$target"
! git -C "$repo" show-ref --verify --quiet "refs/heads/$branch"

git -C "$repo" worktree add -b "$branch" "$target" "$base"

test "$(git -C "$target" rev-parse HEAD)" = "$base"
test -z "$(git -C "$target" status --short)"
printf 'WORKTREE=%s\nBRANCH=%s\nHEAD=%s\n' "$target" "$branch" "$(git -C "$target" rev-parse HEAD)"
echo SZ_PRISM_WORKTREE_CREATED
