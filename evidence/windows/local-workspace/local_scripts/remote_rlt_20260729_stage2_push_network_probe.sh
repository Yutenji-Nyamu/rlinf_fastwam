#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
branch=codex/rlt-pi0-robotwin
expected_head=92e02d9e51c47422696f5ed17a2f15165a6331a6

test "$(git -C "${repo}" rev-parse HEAD)" = "${expected_head}"
test -z "$(git -C "${repo}" status --porcelain --untracked-files=all)"
test "$(git -C "${repo}" rev-list --left-right --count HEAD...@{upstream})" = \
  $'3\t0'

github_code="$(
  timeout 15s curl -L -sS -o /dev/null \
    --connect-timeout 10 --max-time 14 \
    -w '%{http_code}' https://github.com
)" || github_code=000
api_code="$(
  timeout 15s curl -L -sS -o /dev/null \
    --connect-timeout 10 --max-time 14 \
    -w '%{http_code}' https://api.github.com
)" || api_code=000
printf 'github_http\t%s\n' "${github_code}"
printf 'api_http\t%s\n' "${api_code}"

if test "${github_code}" != 200 || test "${api_code}" != 200; then
  printf '%s\n' "NETWORK_PROBE_UNAVAILABLE_NO_PUSH"
  exit 75
fi

timeout 25s git -C "${repo}" push personal "${branch}"
test -z "$(git -C "${repo}" status --porcelain --untracked-files=all)"
test "$(git -C "${repo}" rev-list --left-right --count HEAD...@{upstream})" = \
  $'0\t0'
test "$(
  timeout 15s git -C "${repo}" ls-remote \
    personal "refs/heads/${branch}" | cut -f1
)" = "${expected_head}"
printf '%s\n' STAGE2_DOCS_BOUNDED_PUSH_OK
