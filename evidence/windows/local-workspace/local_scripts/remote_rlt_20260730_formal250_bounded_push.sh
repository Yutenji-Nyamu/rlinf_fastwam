#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export GIT_TERMINAL_PROMPT=0

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
branch=codex/rlt-pi0-robotwin
expected_head=46a2d19b

cd "$repo"
test "$(git branch --show-current)" = "$branch"
test "$(git rev-parse --short=8 HEAD)" = "$expected_head"
test -z "$(git status --porcelain --untracked-files=all)"
printf 'ahead_behind_before\t%s\n' "$(
  git rev-list --left-right --count HEAD...@{upstream}
)"

unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
for url in \
  https://github.com \
  https://api.github.com \
  https://raw.githubusercontent.com
do
  timeout 10s curl -L -sS -o /dev/null \
    --connect-timeout 7 --max-time 9 \
    -w "probe=${url} code=%{http_code} connect=%{time_connect} total=%{time_total}\n" \
    "$url"
done
timeout 15s git ls-remote --heads personal "$branch" >/dev/null

start="$(date +%s)"
timeout 60s git push personal HEAD:"$branch"
end="$(date +%s)"

test -z "$(git status --porcelain --untracked-files=all)"
test "$(git rev-list --left-right --count HEAD...@{upstream})" = $'0\t0'
remote_head="$(
  timeout 15s git ls-remote personal "refs/heads/${branch}" | cut -f1
)"
test "$remote_head" = "$(git rev-parse HEAD)"
printf 'head\t%s\n' "$(git rev-parse HEAD)"
printf 'remote\t%s\n' "$remote_head"
printf 'push_wall_seconds\t%s\n' "$((end - start))"
printf '%s\n' RLT_FORMAL250_PUSH_OK
