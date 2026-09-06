#!/usr/bin/env bash
set -euo pipefail

src=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
target=/root/autodl-tmp/RLinf_rlt_teacher_dvac
branch=codex/rlt-teacher-dvac-weighting
base=2b8199d8ab2e7b110994fd3234bf7007196c3af9

test "$(git -C "$src" rev-parse HEAD)" = "$base"
test -z "$(git -C "$src" status --porcelain)"
test ! -e "$target"
! git -C "$src" show-ref --verify --quiet "refs/heads/$branch"

git -C "$src" worktree add -b "$branch" "$target" "$base"

printf 'CREATED\n'
git -C "$target" rev-parse HEAD
git -C "$target" branch --show-current
git -C "$target" status --short --branch

