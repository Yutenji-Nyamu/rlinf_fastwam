#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
stage=/root/autodl-tmp/qam_formal_docs_sync_20260731_v1
file=docs/rlinf-robotwin-pi0-qam/00_INDEX_AND_IMPLEMENTATION_PLAN.md
branch=codex/qam-pi0-robotwin

test "$(git -C "$repo" rev-parse HEAD)" = \
  cceb4f4404d503503fff2efbca5bf12d6eb17239
test -z "$(git -C "$repo" status --porcelain=v1)"
printf '%s  %s\n' \
  ea20f743f4dad18d15794ca0cc229e9b39bc2bdd45881257708d8da2dc6467b3 \
  "$stage/$file" | sha256sum -c -
install -D -m 0644 "$stage/$file" "$repo/$file"
git -C "$repo" diff --check -- "$file"
if grep -n -F 'NBo7SQqoatnZ' "$repo/$file"; then
  printf 'SECRET_SCAN_FAIL=password\n' >&2
  exit 1
fi
git -C "$repo" add -- "$file"
test "$(git -C "$repo" diff --cached --name-only)" = "$file"
git -C "$repo" diff --cached --check
git -C "$repo" commit -m 'docs(qam): align formal stage status'
printf 'SSOT_COMMIT='
git -C "$repo" rev-parse HEAD
source /etc/network_turbo
GIT_TERMINAL_PROMPT=0 timeout 60 \
  git -C "$repo" push personal "HEAD:$branch"
printf 'AHEAD_BEHIND='
git -C "$repo" rev-list --left-right --count '@{upstream}...HEAD'
printf 'REMOTE='
git -C "$repo" ls-remote --heads personal "$branch"
git -C "$repo" status --short
