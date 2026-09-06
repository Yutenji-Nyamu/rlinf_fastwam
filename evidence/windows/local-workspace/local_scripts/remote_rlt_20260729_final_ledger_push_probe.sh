#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
expected_head=6df42bf488ef10d9c7eb2f89584bc5ab7543a08a

cd "$repo"
test "$(git rev-parse HEAD)" = "$expected_head"
test -z "$(git status --porcelain)"
test "$(git rev-list --left-right --count '@{upstream}...HEAD')" = $'0\t1'

unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
curl --fail --silent --show-error --head \
  --connect-timeout 10 --max-time 20 https://github.com >/dev/null
timeout 25s git push personal codex/rlt-pi0-robotwin

test "$(git rev-list --left-right --count '@{upstream}...HEAD')" = $'0\t0'
remote=$(git ls-remote personal refs/heads/codex/rlt-pi0-robotwin | awk '{print $1}')
test "$remote" = "$expected_head"
printf 'HEAD=%s\nREMOTE=%s\nSTATUS=clean\nUPSTREAM=0/0\n' "$expected_head" "$remote"
