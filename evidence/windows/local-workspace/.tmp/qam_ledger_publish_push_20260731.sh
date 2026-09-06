#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
branch=codex/qam-pi0-robotwin
test "$(git -C "$repo" rev-parse HEAD)" = \
  cceb4f4404d503503fff2efbca5bf12d6eb17239
test -z "$(git -C "$repo" status --porcelain=v1)"
source /etc/network_turbo
GIT_TERMINAL_PROMPT=0 timeout 60 \
  git -C "$repo" push personal "HEAD:$branch"
printf 'AHEAD_BEHIND='
git -C "$repo" rev-list --left-right --count '@{upstream}...HEAD'
printf 'REMOTE='
git -C "$repo" ls-remote --heads personal "$branch"
git -C "$repo" status --short
