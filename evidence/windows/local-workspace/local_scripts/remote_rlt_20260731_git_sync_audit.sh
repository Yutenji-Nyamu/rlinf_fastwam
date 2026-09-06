#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
cd "$repo"

echo "SECTION=IDENTITY"
date --iso-8601=seconds
hostname
id -u

echo "SECTION=REPOSITORY"
printf 'branch='
git branch --show-current
printf 'head='
git rev-parse HEAD
printf 'upstream='
git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}'
echo "STATUS_BEGIN"
git status --short --branch
echo "STATUS_END"
echo "REMOTES_BEGIN"
git remote -v
echo "REMOTES_END"
printf 'ahead_behind='
git rev-list --left-right --count '@{upstream}...HEAD'
echo "AHEAD_COMMITS_BEGIN"
git log --oneline --decorate '@{upstream}..HEAD'
echo "AHEAD_COMMITS_END"
echo "AHEAD_DIFFSTAT_BEGIN"
git diff --stat '@{upstream}..HEAD'
echo "AHEAD_DIFFSTAT_END"
echo "AHEAD_NAMES_BEGIN"
git diff --name-status '@{upstream}..HEAD'
echo "AHEAD_NAMES_END"

echo "SECTION=DOCUMENT_HASHES"
for path in \
  HANDOFF.md \
  docs/rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md \
  docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md \
  docs/rlinf-robotwin-pi0-rltoken/17_STAGE2_FORMAL_RESUME250_TO480_FINAL_RESULT_20260731.md
do
  if [[ -f "$path" ]]; then
    sha256sum "$path"
  else
    printf 'MISSING  %s\n' "$path"
  fi
done

echo "SECTION=NETWORK_CONFIG"
env | grep -iE '^(http|https|all)_proxy=' || true
printf 'git_http_version='
git config --get http.version || printf 'DEFAULT\n'

echo "SECTION=NETWORK_PROBES"
for endpoint in \
  "main https://github.com" \
  "api https://api.github.com" \
  "raw https://raw.githubusercontent.com"
do
  name=${endpoint%% *}
  url=${endpoint#* }
  curl -L -sS -o /dev/null --connect-timeout 7 --max-time 10 \
    -w "$name code=%{http_code} connect=%{time_connect} total=%{time_total}\n" \
    "$url" || true
done
echo "LS_REMOTE_BEGIN"
set +e
GIT_TERMINAL_PROMPT=0 timeout 15 \
  git ls-remote --heads personal codex/rlt-pi0-robotwin
ls_remote_rc=$?
set -e
printf 'ls_remote_rc=%s\n' "$ls_remote_rc"
echo "LS_REMOTE_END"

echo "RLT_GIT_SYNC_AUDIT_DONE"
