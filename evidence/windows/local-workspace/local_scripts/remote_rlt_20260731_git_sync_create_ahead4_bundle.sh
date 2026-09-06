#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
output=/root/autodl-tmp/experiment_exports/rlt_git_sync_20260731_0957_v1/codex-rlt-ahead4.bundle
branch=codex/rlt-pi0-robotwin
expected_head=2b8199d8ab2e7b110994fd3234bf7007196c3af9
base=personal/codex/rlt-pi0-robotwin

cd "$repo"
test "$(git rev-parse HEAD)" = "$expected_head"
test -z "$(git status --porcelain)"
test "$(git rev-list --left-right --count "$base...HEAD")" = $'0\t4'
test ! -e "$output"

git bundle create "$output" "$branch" "^$base"
git bundle verify "$output"
stat -c 'bundle=%n bytes=%s mtime=%y' "$output"
sha256sum "$output"
printf 'BUNDLE_HEADS_BEGIN\n'
git bundle list-heads "$output"
printf 'BUNDLE_HEADS_END\n'
printf 'RLT_GIT_SYNC_AHEAD4_BUNDLE_READY\n'
