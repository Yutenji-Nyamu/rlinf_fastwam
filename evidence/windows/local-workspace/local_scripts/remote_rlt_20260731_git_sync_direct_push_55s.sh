#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
branch=codex/rlt-pi0-robotwin
expected_head=2b8199d8ab2e7b110994fd3234bf7007196c3af9
upstream=personal/codex/rlt-pi0-robotwin

cd "$repo"
test "$(git branch --show-current)" = "$branch"
test "$(git rev-parse HEAD)" = "$expected_head"
test -z "$(git status --porcelain)"
test "$(git rev-list --left-right --count "$upstream...HEAD")" = $'0\t4'

set +e
timeout 55 git push personal \
  "refs/heads/$branch:refs/heads/$branch"
push_rc=$?
set -e
printf 'PUSH_RC=%s\n' "$push_rc"
test "$push_rc" -eq 0

test "$(git rev-parse "$upstream")" = "$expected_head"
test "$(git rev-list --left-right --count "$upstream...HEAD")" = $'0\t0'
test -z "$(git status --porcelain)"
printf 'PUSH_OK_HEAD=%s\n' "$expected_head"
