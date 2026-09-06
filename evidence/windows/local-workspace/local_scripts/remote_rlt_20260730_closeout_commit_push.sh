#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
stage=/root/autodl-tmp/tmp/rlt_stage2_fresh_closeout_20260730_v3
manifest="${stage}/UPLOAD_SHA256SUMS"
expected_head=6fd3ee7106fb82f06eda82603c41a09767151709
expected_manifest_sha=d48f1bafb564f3877b31b8891984bb5faa45c568252cc2a9bbad393aa507a54c

cd "${repo}"
test "$(git branch --show-current)" = codex/rlt-pi0-robotwin
test "$(git rev-parse HEAD)" = "${expected_head}"
test "$(sha256sum "${manifest}" | cut -d' ' -f1)" = "${expected_manifest_sha}"
read -r left right < <(git rev-list --left-right --count '@{upstream}...HEAD')
test "${left}" = 0
test "${right}" = 0

mapfile -t paths < <(cut -d' ' -f3- "${manifest}")
test "${#paths[@]}" = 21
git add -f -- "${paths[@]}"
git diff --cached --name-only | LC_ALL=C sort \
  > /tmp/rlt_closeout_staged_paths.txt
printf '%s\n' "${paths[@]}" | LC_ALL=C sort \
  > /tmp/rlt_closeout_expected_paths.txt
cmp /tmp/rlt_closeout_staged_paths.txt /tmp/rlt_closeout_expected_paths.txt

# Machine logs/CSV preserve original console bytes; text sources get whitespace gate.
git diff --cached --check -- \
  HANDOFF.md \
  'docs/rlinf-robotwin-pi0-rltoken/*.md' \
  'docs/rlinf-robotwin-pi0-rltoken/evidence/*.md' \
  'docs/rlinf-robotwin-pi0-rltoken/evidence/stage2_pre_smoke_20260729/README.md' \
  'docs/rlinf-robotwin-pi0-rltoken/evidence/stage2_fresh_smoke_20260730/*.md' \
  'docs/rlinf-robotwin-pi0-rltoken/evidence/stage2_fresh_smoke_20260730/*.json' \
  'docs/rlinf-robotwin-pi0-rltoken/evidence/stage2_fresh_smoke_20260730/*.yaml' \
  'docs/rlinf-robotwin-pi0-rltoken/evidence/stage2_fresh_smoke_20260730/*.tsv'

git commit -m "docs(rlt): record Stage 2 fresh smoke"
new_head="$(git rev-parse HEAD)"
test -z "$(git status --short)"

main_code="$(
  curl -L -sS -o /dev/null \
    --connect-timeout 7 --max-time 10 \
    -w '%{http_code}' https://github.com || true
)"
test "${main_code}" = 200
GIT_TERMINAL_PROMPT=0 timeout 60 \
  git push personal HEAD:codex/rlt-pi0-robotwin
remote_head="$(
  timeout 15 git ls-remote \
    personal refs/heads/codex/rlt-pi0-robotwin \
    | awk '{print $1}'
)"
test "${remote_head}" = "${new_head}"
read -r left_after right_after < <(
  git rev-list --left-right --count '@{upstream}...HEAD'
)
test "${left_after}" = 0
test "${right_after}" = 0

printf 'commit\t%s\n' "${new_head}"
printf 'remote_head\t%s\n' "${remote_head}"
printf 'left_right_after\t%s/%s\n' "${left_after}" "${right_after}"
printf '%s\n' RLT_STAGE2_FRESH_CLOSEOUT_COMMIT_PUSH_OK
