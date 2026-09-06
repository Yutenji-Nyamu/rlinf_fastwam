#!/usr/bin/env bash
set -euo pipefail
repo=/root/autodl-tmp/RLinf_rlt_dvac_pure
branch=codex/rlt-dvac-pure-reference-bc
expected=cb88e9c5

test "$(git -C "$repo" rev-parse --short=8 HEAD)" = "$expected"
test -z "$(git -C "$repo" status --short)"
(
  source /etc/network_turbo >/dev/null 2>&1
  before=$(timeout 20 git -C "$repo" ls-remote personal "refs/heads/$branch" | awk '{print $1}')
  echo "remote_before=${before:-absent}"
  if test -z "$before"; then
    timeout 60 env GIT_TERMINAL_PROMPT=0 git -C "$repo" push personal "HEAD:refs/heads/$branch"
  fi
  after=$(timeout 20 git -C "$repo" ls-remote personal "refs/heads/$branch" | awk '{print $1}')
  echo "remote_after=$after"
  test "$after" = "$(git -C "$repo" rev-parse HEAD)"
)
env | grep -iE '^(http|https|all)_proxy=' || true
