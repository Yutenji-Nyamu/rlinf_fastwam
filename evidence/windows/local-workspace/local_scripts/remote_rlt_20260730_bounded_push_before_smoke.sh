#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export GIT_TERMINAL_PROMPT=0

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
branch=codex/rlt-pi0-robotwin
expected_head=6fd3ee7106fb82f06eda82603c41a09767151709

test "$(git -C "${repo}" rev-parse HEAD)" = "${expected_head}"
test -z "$(git -C "${repo}" status --porcelain --untracked-files=all)"
test "$(git -C "${repo}" rev-list --left-right --count HEAD...@{upstream})" = \
  $'5\t0'
test "$(
  timeout 10s curl -L -sS -o /dev/null \
    --connect-timeout 7 --max-time 9 \
    -w '%{http_code}' https://github.com
)" = 200

start="$(date +%s)"
timeout 60s git -C "${repo}" push personal "${branch}"
end="$(date +%s)"

test -z "$(git -C "${repo}" status --porcelain --untracked-files=all)"
test "$(git -C "${repo}" rev-list --left-right --count HEAD...@{upstream})" = \
  $'0\t0'
remote_head="$(
  timeout 15s git -C "${repo}" ls-remote \
    personal "refs/heads/${branch}" | cut -f1
)"
test "${remote_head}" = "${expected_head}"
printf 'head\t%s\n' "${expected_head}"
printf 'remote\t%s\n' "${remote_head}"
printf 'push_wall_seconds\t%s\n' "$((end - start))"
printf '%s\n' RLT_PRE_SMOKE_PUSH_OK
