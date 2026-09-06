#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
branch=codex/rlt-pi0-robotwin
expected_head=92e02d9e51c47422696f5ed17a2f15165a6331a6

test "$(git -C "${repo}" branch --show-current)" = "${branch}"
test "$(git -C "${repo}" rev-parse HEAD)" = "${expected_head}"
test -z "$(git -C "${repo}" status --porcelain --untracked-files=all)"
if pgrep -af 'git .*push.*codex/rlt-pi0-robotwin' \
  | grep -v -E 'pgrep -af|retry_stage2_docs_push' >/dev/null
then
  printf '%s\n' "A matching push process is still active." >&2
  exit 1
fi

printf 'before_left_right\t%s\n' "$(
  git -C "${repo}" rev-list --left-right --count HEAD...@{upstream}
)"
remote_head="$(
  timeout 30s git -C "${repo}" ls-remote \
    personal "refs/heads/${branch}" | cut -f1
)" || remote_head=
printf 'remote_probe\t%s\n' "${remote_head:-UNAVAILABLE}"

if test "${remote_head}" != "${expected_head}"; then
  timeout 240s git -C "${repo}" push personal "${branch}"
else
  timeout 120s git -C "${repo}" fetch personal "${branch}"
fi

remote_head="$(
  timeout 30s git -C "${repo}" ls-remote \
    personal "refs/heads/${branch}" | cut -f1
)"
test "${remote_head}" = "${expected_head}"
test -z "$(git -C "${repo}" status --porcelain --untracked-files=all)"
test "$(git -C "${repo}" rev-list --left-right --count HEAD...@{upstream})" = \
  $'0\t0'
printf 'head\t%s\n' "${expected_head}"
printf 'remote\t%s\n' "${remote_head}"
printf '%s\n' STAGE2_DOCS_PUSH_CONFIRMED
