#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin
expected_patch_sha=dae5cbb31d01a6a114f91b34c9e19ecf01b341d03eb01cb57d912a79ec3299d0

test "$(git -C "$repo" rev-parse HEAD)" = \
  6d0db56bf26f972cd27fa29535f5eb939e80e5bf
test "$(git -C "$repo" branch --show-current)" = \
  codex/ogpo-pi0-robotwin
test -z "$(git -C "$repo" diff --name-only)"
test -z "$(git -C "$repo" ls-files --others --exclude-standard)"
actual_patch_sha=$(
  git -C "$repo" diff --cached --binary --full-index HEAD \
    | sha256sum \
    | awk '{print $1}'
)
test "$actual_patch_sha" = "$expected_patch_sha"

GIT_TERMINAL_PROMPT=0 git -C "$repo" commit \
  -m 'feat(embodiment): add pi0 OGPO+CA for RoboTwin'
printf 'COMMIT_HEAD=%s\n' "$(git -C "$repo" rev-parse HEAD)"
git -C "$repo" show --stat --oneline --decorate --no-renames HEAD
test -z "$(git -C "$repo" status --porcelain=v1 --untracked-files=all)"
printf 'COMMIT_WORKTREE_CLEAN\n'
