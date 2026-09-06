#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
output=/root/autodl-tmp/experiment_exports/rlt_git_sync_20260731_0957_v1/codex-rlt-ahead3.bundle
branch=codex/rlt-pi0-robotwin
expected_head=d81267b57eb5ce13e6452139aaeba02af3911624
base=personal/codex/rlt-pi0-robotwin

cd "$repo"
[[ "$(git rev-parse HEAD)" == "$expected_head" ]]
[[ -z "$(git status --short)" ]]
[[ "$(git rev-list --left-right --count "$base...HEAD")" == $'0\t3' ]]
[[ ! -e "$output" ]]

git bundle create "$output" "$branch" "^$base"
git bundle verify "$output"
stat -c 'bundle=%n bytes=%s mtime=%y' "$output"
sha256sum "$output"
echo "BUNDLE_HEADS_BEGIN"
git bundle list-heads "$output"
echo "BUNDLE_HEADS_END"
echo "RLT_GIT_SYNC_BUNDLE_READY"
