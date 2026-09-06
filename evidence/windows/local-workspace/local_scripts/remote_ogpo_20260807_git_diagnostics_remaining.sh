#!/usr/bin/env bash
set -u

repo=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin

test "$(git -C "$repo" rev-parse HEAD)" = \
  5d5c84e3ac4efa1713a4139a05ac1b776e634ed3 || exit 1
test -z "$(git -C "$repo" status --porcelain=v1 --untracked-files=all)" || exit 1

curl -L -sS -o /dev/null --connect-timeout 7 --max-time 10 \
  -w 'api code=%{http_code} connect=%{time_connect} total=%{time_total}\n' \
  https://api.github.com
printf 'API_EXIT=%s\n' "$?"
curl -L -sS -o /dev/null --connect-timeout 7 --max-time 10 \
  -w 'raw code=%{http_code} connect=%{time_connect} total=%{time_total}\n' \
  https://raw.githubusercontent.com
printf 'RAW_EXIT=%s\n' "$?"
printf 'REMOTE_BRANCH_BEFORE_PUSH\n'
GIT_TERMINAL_PROMPT=0 timeout 15 git -C "$repo" ls-remote \
  --heads personal codex/ogpo-pi0-robotwin
printf 'LS_REMOTE_EXIT=%s\n' "$?"
