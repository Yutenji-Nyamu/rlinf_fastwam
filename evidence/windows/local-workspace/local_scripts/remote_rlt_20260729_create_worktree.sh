#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf
worktree=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
branch=codex/rlt-pi0-robotwin
base=48a775db09c16c455aeba7b0600c920e7c80d534

git -C "$repo" cat-file -e "$base^{commit}"
if test -e "$worktree"; then
  echo "refusing: target path already exists: $worktree" >&2
  exit 10
fi
if git -C "$repo" show-ref --verify --quiet "refs/heads/$branch"; then
  echo "refusing: local branch already exists: $branch" >&2
  exit 11
fi

git -C "$repo" worktree add -b "$branch" "$worktree" "$base"

echo '[created]'
git -C "$worktree" branch --show-current
git -C "$worktree" rev-parse HEAD
git -C "$worktree" status --short
git -C "$repo" worktree list --porcelain
