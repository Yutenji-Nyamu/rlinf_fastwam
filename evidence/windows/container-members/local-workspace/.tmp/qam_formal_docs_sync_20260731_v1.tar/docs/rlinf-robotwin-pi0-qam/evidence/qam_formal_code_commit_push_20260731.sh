#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
branch=codex/qam-pi0-robotwin
files=(
  examples/embodiment/config/robotwin_adjust_bottle_qam_openpi.yaml
  rlinf/config.py
  rlinf/workers/actor/fsdp_qam_policy_worker.py
  tests/workers/test_qam_worker_helpers.py
)

cd "$repo"
test "$(git branch --show-current)" = "$branch"
test "$(git rev-parse HEAD)" = ced8672f322187b71939bd2859842619c6284d05
expected_dirty=$(cat <<'EOF'
 M examples/embodiment/config/robotwin_adjust_bottle_qam_openpi.yaml
 M rlinf/config.py
 M rlinf/workers/actor/fsdp_qam_policy_worker.py
 M tests/workers/test_qam_worker_helpers.py
EOF
)
test "$(git status --porcelain=v1)" = "$expected_dirty"

git diff --check -- "${files[@]}"
git add -- "${files[@]}"
git diff --cached --check
git diff --cached --stat
git commit -m "feat(qam): schedule in-process AM activation"

env | grep -iE '^(http|https|all)_proxy=' || true
git config --get http.version || printf 'DEFAULT\n'
curl -L -sS -o /dev/null --connect-timeout 7 --max-time 10 \
  -w 'main code=%{http_code} connect=%{time_connect} total=%{time_total}\n' \
  https://github.com
timeout 15 git ls-remote --heads personal "$branch"
git rev-list --left-right --count '@{upstream}...HEAD'
GIT_TERMINAL_PROMPT=0 timeout 60 git push personal "HEAD:$branch"
git rev-list --left-right --count '@{upstream}...HEAD'
git ls-remote --heads personal "$branch"
test -z "$(git status --porcelain=v1)"
