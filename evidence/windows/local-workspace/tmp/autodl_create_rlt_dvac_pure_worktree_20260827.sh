#!/usr/bin/env bash
set -euo pipefail
source_repo=/root/autodl-tmp/RLinf_rlt_dvac_success_bc
target=/root/autodl-tmp/RLinf_rlt_dvac_pure
branch=codex/rlt-dvac-pure-reference-bc

test ! -e "$target"
git -C "$source_repo" worktree add -b "$branch" "$target" 848b61278687702ea717c56b3734f1486cea3b95
git -C "$target" status --short --branch
git -C "$target" rev-parse HEAD
