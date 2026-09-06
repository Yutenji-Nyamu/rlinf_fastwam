#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
branch=codex/qam-pi0-robotwin
expected_head=d6fa0f0f4915587ae5e6a03c580fea7938acd3ca

test "$(git -C "$repo" rev-parse HEAD)" = "$expected_head"
test -z "$(git -C "$repo" status --porcelain=v1)"
env | grep -iE '^(http|https|all)_proxy=' || true
git -C "$repo" config --get http.version || printf 'DEFAULT\n'

set +e
curl -L -sS -o /dev/null --connect-timeout 7 --max-time 10 \
  -w 'main code=%{http_code} connect=%{time_connect} total=%{time_total}\n' \
  https://github.com
main_status=$?
timeout 15 git -C "$repo" ls-remote --heads personal "$branch"
ls_status=$?
set -e

if test "$main_status" -eq 0 && test "$ls_status" -eq 0; then
  printf 'PUSH_ROUTE=direct\n'
  GIT_TERMINAL_PROMPT=0 timeout 60 \
    git -C "$repo" push personal "HEAD:$branch"
else
  printf 'PUSH_ROUTE=temporary_network_turbo main_status=%s ls_status=%s\n' \
    "$main_status" "$ls_status"
  source /etc/network_turbo
  GIT_TERMINAL_PROMPT=0 timeout 60 \
    git -C "$repo" push personal "HEAD:$branch"
fi

printf 'AHEAD_BEHIND='
git -C "$repo" rev-list --left-right --count '@{upstream}...HEAD'
printf 'REMOTE='
git -C "$repo" ls-remote --heads personal "$branch"
printf 'STATUS\n'
git -C "$repo" status --short
