#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
stage=/root/autodl-tmp/experiment_exports/rlt_git_sync_20260731_0957_v1/final_sync_docs_staging
expected_head=2b8199d8ab2e7b110994fd3234bf7007196c3af9

test "$(git -C "$repo" branch --show-current)" = codex/rlt-pi0-robotwin
test "$(git -C "$repo" rev-parse HEAD)" = "$expected_head"
test "$(git -C "$repo" rev-parse personal/codex/rlt-pi0-robotwin)" = "$expected_head"
test "$(git -C "$repo" rev-list --left-right --count \
  personal/codex/rlt-pi0-robotwin...HEAD)" = $'0\t0'
test -z "$(git -C "$repo" status --porcelain)"
test ! -e "$stage"

mkdir -p "$stage/docs/rlinf-robotwin-pi0-rltoken/evidence"
printf 'STAGE=%s\n' "$stage"
printf 'HEAD=%s\n' "$expected_head"
printf 'UPSTREAM_SYNCED=YES\n'
