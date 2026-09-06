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

main_code="$(
  curl -L -sS -o /dev/null \
    --connect-timeout 5 --max-time 12 \
    -w '%{http_code}' https://github.com/ || true
)"
printf 'GITHUB_MAIN_HTTP=%s\n' "$main_code"

if ! timeout 20 git ls-remote personal \
  "refs/heads/$branch" > /tmp/rlt_ahead4_ls_remote.txt
then
  printf 'PUSH_SKIPPED_NETWORK_UNAVAILABLE\n'
  exit 75
fi

remote_head="$(cut -f1 /tmp/rlt_ahead4_ls_remote.txt)"
test "$remote_head" = 9bb2dd78feff7133780c3df6a88618d10168c4e4

timeout 240 git push personal \
  "refs/heads/$branch:refs/heads/$branch"

test "$(git rev-parse "$upstream")" = "$expected_head"
test "$(git rev-list --left-right --count "$upstream...HEAD")" = $'0\t0'
test -z "$(git status --porcelain)"
printf 'PUSH_OK_HEAD=%s\n' "$expected_head"
