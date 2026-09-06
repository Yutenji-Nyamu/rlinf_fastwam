#!/usr/bin/env bash
set -euo pipefail

root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-observe-7d07a421
base=7d07a4212ee6858cc333e1d4fab7a37256d1f839
branch=codex/sz-current-pi0-dvac-observe
subject='feat: add opt-in pi0 DVAC telemetry'

cd "$root"
test "$(git branch --show-current)" = "$branch"
identity_source=61996e15cc7f5a32bd6012b61b20893d94636c82
export GIT_AUTHOR_NAME="$(git show -s --format=%an "$identity_source")"
export GIT_AUTHOR_EMAIL="$(git show -s --format=%ae "$identity_source")"
export GIT_COMMITTER_NAME="$(git show -s --format=%cn "$identity_source")"
export GIT_COMMITTER_EMAIL="$(git show -s --format=%ce "$identity_source")"
test -n "$GIT_AUTHOR_NAME"
test -n "$GIT_AUTHOR_EMAIL"
test -n "$GIT_COMMITTER_NAME"
test -n "$GIT_COMMITTER_EMAIL"
echo "GIT_IDENTITY_SOURCE=$identity_source"
echo "GIT_AUTHOR_NAME=$GIT_AUTHOR_NAME"

head=$(git rev-parse HEAD)
if test "$head" = "$base"; then
  test -z "$(git diff --name-only)"
  test -z "$(git diff --name-only --diff-filter=U)"
  git diff --cached --check
  mapfile -t staged < <(git diff --cached --name-only | sort)
  expected=(
    evaluations/robotwin/robotwin_adjust_bottle_openpi_dvac_eval.yaml
    rlinf/models/embodiment/openpi/openpi_action_model.py
    rlinf/utils/dvac_telemetry.py
    rlinf/workers/env/env_worker.py
    rlinf/workers/rollout/hf/huggingface_worker.py
    tests/unit_tests/test_dvac_telemetry.py
  )
  test "${#staged[@]}" -eq "${#expected[@]}"
  for idx in "${!expected[@]}"; do
    test "${staged[$idx]}" = "${expected[$idx]}"
  done
  git commit -m "$subject"
else
  test "$(git rev-parse HEAD^)" = "$base"
  test "$(git log -1 --format=%s)" = "$subject"
  test -z "$(git status --porcelain)"
fi

commit=$(git rev-parse HEAD)
test "$(git rev-parse HEAD^)" = "$base"
test "$(git log -1 --format=%s)" = "$subject"
test -z "$(git status --porcelain)"

export GIT_TERMINAL_PROMPT=0
timeout --signal=INT --kill-after=5s 120s \
  git push --set-upstream personal "HEAD:refs/heads/$branch"

remote_head=$(git ls-remote personal "refs/heads/$branch" | awk '{print $1}')
test "$remote_head" = "$commit"
test "$(git rev-parse '@{upstream}')" = "$commit"
test -z "$(git status --porcelain)"

echo "COMMIT=$commit"
echo "REMOTE_HEAD=$remote_head"
echo "UPSTREAM=$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}')"
git status --short --branch
