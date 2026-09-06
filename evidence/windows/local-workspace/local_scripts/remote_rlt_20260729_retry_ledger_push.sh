#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
expected_head=6df42bf488ef10d9c7eb2f89584bc5ab7543a08a

cd "$repo"
test "$(git rev-parse HEAD)" = "$expected_head"
test -z "$(git status --porcelain)"
test "$(git rev-list --left-right --count '@{upstream}...HEAD')" = $'0\t1'

timeout 240s git push personal codex/rlt-pi0-robotwin

test "$(git rev-list --left-right --count '@{upstream}...HEAD')" = $'0\t0'
remote=$(git ls-remote personal refs/heads/codex/rlt-pi0-robotwin | awk '{print $1}')
test "$remote" = "$expected_head"
printf 'HEAD=%s\nREMOTE=%s\nSTATUS=clean\nUPSTREAM=0/0\n' "$expected_head" "$remote"
