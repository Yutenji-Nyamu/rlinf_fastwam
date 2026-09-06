#!/usr/bin/env bash

set -euo pipefail
export GIT_OPTIONAL_LOCKS=0

repo=/root/autodl-tmp/RLinf
target=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin
branch=codex/idea2-dvac-pi0-robotwin
base=6d0db56bf26f972cd27fa29535f5eb939e80e5bf

date --iso-8601=seconds 2>/dev/null || date
hostname
pwd
id -u

test ! -e "$target"
if git -C "$repo" show-ref --verify --quiet "refs/heads/$branch"; then
  printf 'target branch already exists: %s\n' "$branch" >&2
  exit 2
fi
git -C "$repo" cat-file -e "${base}^{commit}"
git -C "$repo" worktree add -b "$branch" "$target" "$base"

printf '\ncreated_state:\n'
git -C "$target" rev-parse HEAD
git -C "$target" symbolic-ref --quiet --short HEAD
git -C "$target" status --short
git -C "$repo" worktree list --porcelain

