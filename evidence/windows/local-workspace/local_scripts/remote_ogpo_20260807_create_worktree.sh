#!/usr/bin/env bash
set -euo pipefail

base=/root/autodl-tmp/RLinf
target=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin
branch=codex/ogpo-pi0-robotwin
expected=6d0db56bf26f972cd27fa29535f5eb939e80e5bf

echo '=== preconditions ==='
actual=$(git -C "$base" rev-parse HEAD)
printf 'base_head=%s\n' "$actual"
test "$actual" = "$expected"
test ! -e "$target"
if git -C "$base" show-ref --verify --quiet "refs/heads/$branch"; then
  echo "branch already exists: $branch" >&2
  exit 2
fi

echo '=== create-worktree ==='
git -C "$base" worktree add -b "$branch" "$target" "$expected"

echo '=== result ==='
git -C "$target" rev-parse HEAD
git -C "$target" branch --show-current
git -C "$target" status --short --branch
git -C "$base" worktree list --porcelain
