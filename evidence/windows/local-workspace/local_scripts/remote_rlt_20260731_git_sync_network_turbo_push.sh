#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
branch=codex/rlt-pi0-robotwin
expected_head=2b8199d8ab2e7b110994fd3234bf7007196c3af9
expected_remote=9bb2dd78feff7133780c3df6a88618d10168c4e4
upstream=personal/codex/rlt-pi0-robotwin

test -r /etc/network_turbo
cd "$repo"
test "$(git branch --show-current)" = "$branch"
test "$(git rev-parse HEAD)" = "$expected_head"
test -z "$(git status --porcelain)"
test "$(git rev-parse "$upstream")" = "$expected_remote"
test "$(git rev-list --left-right --count "$upstream...HEAD")" = $'0\t4'

(
  set +u
  # Do not print or persist proxy endpoints from the provider-owned script.
  source /etc/network_turbo >/dev/null 2>&1
  set -u

  for name in http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
  do
    if [[ -n "${!name:-}" ]]; then
      printf '%s=SET\n' "$name"
    else
      printf '%s=UNSET\n' "$name"
    fi
  done

  timeout 15 git ls-remote personal \
    "refs/heads/$branch" > /tmp/rlt_turbo_ls_remote.txt
  remote_head="$(cut -f1 /tmp/rlt_turbo_ls_remote.txt)"
  printf 'TURBO_LS_REMOTE_HEAD=%s\n' "$remote_head"
  test "$remote_head" = "$expected_remote"

  GIT_TERMINAL_PROMPT=0 timeout 40 git push personal \
    "refs/heads/$branch:refs/heads/$branch"
)

test "$(git rev-parse "$upstream")" = "$expected_head"
test "$(git rev-list --left-right --count "$upstream...HEAD")" = $'0\t0'
test -z "$(git status --porcelain)"
printf 'TURBO_PUSH_OK_HEAD=%s\n' "$expected_head"
