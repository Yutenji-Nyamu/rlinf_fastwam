#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin
branch=codex/ogpo-pi0-robotwin
expected_head=5d5c84e3ac4efa1713a4139a05ac1b776e634ed3

test -r /etc/network_turbo
test "$(git -C "$repo" branch --show-current)" = "$branch"
test "$(git -C "$repo" rev-parse HEAD)" = "$expected_head"
test -z "$(git -C "$repo" status --porcelain=v1 --untracked-files=all)"

for name in http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
do
  test -z "${!name:-}"
done

(
  set +u
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

  remote_line=$(
    GIT_TERMINAL_PROMPT=0 timeout 15 git -C "$repo" ls-remote personal \
      "refs/heads/$branch"
  )
  remote_head=${remote_line%%$'\t'*}
  printf 'TURBO_REMOTE_HEAD_BEFORE=%s\n' "${remote_head:-ABSENT}"
  if [[ -n "$remote_head" && "$remote_head" != "$expected_head" ]]; then
    printf 'UNEXPECTED_REMOTE_HEAD=%s\n' "$remote_head" >&2
    exit 3
  fi

  if [[ -z "$remote_head" ]]; then
    GIT_TERMINAL_PROMPT=0 timeout 60 git -C "$repo" push -u personal \
      "HEAD:refs/heads/$branch"
  fi

  post_line=$(
    GIT_TERMINAL_PROMPT=0 timeout 15 git -C "$repo" ls-remote personal \
      "refs/heads/$branch"
  )
  post_head=${post_line%%$'\t'*}
  test "$post_head" = "$expected_head"
  printf 'TURBO_REMOTE_HEAD_AFTER=%s\n' "$post_head"
)

for name in http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
do
  test -z "${!name:-}"
done
test "$(git -C "$repo" rev-parse '@{upstream}')" = "$expected_head"
test "$(git -C "$repo" rev-list --left-right --count '@{upstream}...HEAD')" = $'0\t0'
test -z "$(git -C "$repo" status --porcelain=v1 --untracked-files=all)"
printf 'TURBO_PUSH_OK head=%s ahead_behind=0/0 worktree=clean\n' "$expected_head"
