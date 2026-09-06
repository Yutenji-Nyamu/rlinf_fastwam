#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
stage=/root/autodl-tmp/tmp/rlt_stage2_docs_upload_20260729_v2
code_commit=3b610cb4685a1d41c97da64df67ab86561697dfd

cd "${repo}"
test "$(git branch --show-current)" = codex/rlt-pi0-robotwin
git merge-base --is-ancestor "${code_commit}" HEAD
git diff --quiet \
  "${code_commit}" -- . ':(exclude)docs/**' ':(exclude)HANDOFF.md'
test -f "${stage}/UPLOAD_SHA256SUMS"

expected_list="$(
  sed 's/^[0-9a-f]\{64\}  //' "${stage}/UPLOAD_SHA256SUMS" | LC_ALL=C sort
)"
actual_list="$(
  git status --porcelain --untracked-files=all | cut -c4- | LC_ALL=C sort
)"
test -n "${actual_list}"
unexpected="$(
  comm -23 \
    <(printf '%s\n' "${actual_list}") \
    <(printf '%s\n' "${expected_list}")
)"
test -z "${unexpected}"

mapfile -t paths < <(
  sed 's/^[0-9a-f]\{64\}  //' "${stage}/UPLOAD_SHA256SUMS"
)
git add -f -- "${paths[@]}"
test "$(git diff --cached --name-only | LC_ALL=C sort)" = "${expected_list}"
git diff --cached --check
git commit -m "docs(rlt): prepare Stage 2 smoke approval"

commit="$(git rev-parse HEAD)"
printf 'commit\t%s\n' "${commit}"
timeout 60s git push personal codex/rlt-pi0-robotwin
test -z "$(git status --porcelain --untracked-files=all)"
test "$(git rev-list --left-right --count HEAD...@{upstream})" = $'0\t0'
printf 'remote\t%s\n' "$(
  git ls-remote personal refs/heads/codex/rlt-pi0-robotwin | cut -f1
)"
printf '%s\n' "STAGE2_DOCS_COMMIT_PUSH_OK"
