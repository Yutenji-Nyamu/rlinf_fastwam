#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
stage=/root/autodl-tmp/experiment_exports/rlt_git_sync_20260731_0957_v1/authority_docs_staging
expected_head=d81267b57eb5ce13e6452139aaeba02af3911624

test "$(git -C "$repo" branch --show-current)" = codex/rlt-pi0-robotwin
test "$(git -C "$repo" rev-parse HEAD)" = "$expected_head"
test -z "$(git -C "$repo" status --porcelain)"
test ! -e "$stage"

mkdir -p "$stage/docs/rlinf-robotwin-pi0-rltoken/evidence"
printf 'STAGE=%s\n' "$stage"
printf 'HEAD=%s\n' "$expected_head"
printf 'WORKTREE=CLEAN\n'
