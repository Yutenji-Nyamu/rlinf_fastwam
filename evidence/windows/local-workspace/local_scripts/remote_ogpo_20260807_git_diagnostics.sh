#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin

test "$(git -C "$repo" rev-parse HEAD)" = \
  5d5c84e3ac4efa1713a4139a05ac1b776e634ed3
test "$(git -C "$repo" branch --show-current)" = \
  codex/ogpo-pi0-robotwin
test -z "$(git -C "$repo" status --porcelain=v1 --untracked-files=all)"
git -C "$repo" remote get-url personal >/dev/null

printf 'PROXY_ENV_NAMES\n'
env | sed -n -E 's/^((http|https|all)_proxy)=.*/\1=PRESENT/Ip'
printf 'GIT_HTTP_VERSION='
git -C "$repo" config --get http.version || printf 'DEFAULT\n'
curl -L -sS -o /dev/null --connect-timeout 7 --max-time 10 \
  -w 'main code=%{http_code} connect=%{time_connect} total=%{time_total}\n' \
  https://github.com
curl -L -sS -o /dev/null --connect-timeout 7 --max-time 10 \
  -w 'api code=%{http_code} connect=%{time_connect} total=%{time_total}\n' \
  https://api.github.com
curl -L -sS -o /dev/null --connect-timeout 7 --max-time 10 \
  -w 'raw code=%{http_code} connect=%{time_connect} total=%{time_total}\n' \
  https://raw.githubusercontent.com
printf 'REMOTE_BRANCH_BEFORE_PUSH\n'
GIT_TERMINAL_PROMPT=0 timeout 15 git -C "$repo" ls-remote \
  --heads personal codex/ogpo-pi0-robotwin
