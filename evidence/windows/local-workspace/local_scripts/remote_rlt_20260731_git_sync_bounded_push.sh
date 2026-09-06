#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
expected_head=d81267b57eb5ce13e6452139aaeba02af3911624
branch=codex/rlt-pi0-robotwin
cd "$repo"

[[ "$(git branch --show-current)" == "$branch" ]]
[[ "$(git rev-parse HEAD)" == "$expected_head" ]]
[[ -z "$(git status --short)" ]]
[[ "$(git rev-list --left-right --count '@{upstream}...HEAD')" == $'0\t3' ]]

echo "NETWORK_PRECHECK_BEGIN"
main_code="$(
  curl -L -sS -o /dev/null --connect-timeout 7 --max-time 10 \
    -w '%{http_code}' https://github.com || true
)"
printf 'main_code=%s\n' "$main_code"
set +e
remote_before="$(
  GIT_TERMINAL_PROMPT=0 timeout 15 \
    git ls-remote --heads personal "$branch"
)"
ls_remote_rc=$?
set -e
printf 'ls_remote_rc=%s\n' "$ls_remote_rc"
printf 'remote_before=%s\n' "$remote_before"
echo "NETWORK_PRECHECK_END"

if [[ "$main_code" != "200" || "$ls_remote_rc" != "0" ]]; then
  echo "PUSH_SKIPPED_NETWORK_UNAVAILABLE"
  exit 75
fi

echo "PUSH_BEGIN"
start_epoch=$(date +%s)
set +e
GIT_TERMINAL_PROMPT=0 timeout 60 \
  git push personal HEAD:"$branch"
push_rc=$?
set -e
end_epoch=$(date +%s)
printf 'push_rc=%s elapsed_seconds=%s\n' "$push_rc" "$((end_epoch - start_epoch))"
echo "PUSH_END"
[[ "$push_rc" == "0" ]]

remote_after="$(
  GIT_TERMINAL_PROMPT=0 timeout 15 \
    git ls-remote --heads personal "$branch"
)"
remote_after_head=$(awk '{print $1}' <<<"$remote_after")
[[ "$remote_after_head" == "$expected_head" ]]
[[ "$(git rev-list --left-right --count '@{upstream}...HEAD')" == $'0\t0' ]]
[[ -z "$(git status --short)" ]]

printf 'local_head=%s\n' "$(git rev-parse HEAD)"
printf 'remote_after=%s\n' "$remote_after"
printf 'ahead_behind=%s\n' "$(
  git rev-list --left-right --count '@{upstream}...HEAD'
)"
git status --short --branch
echo "RLT_GIT_SYNC_PUSH_PASS"
