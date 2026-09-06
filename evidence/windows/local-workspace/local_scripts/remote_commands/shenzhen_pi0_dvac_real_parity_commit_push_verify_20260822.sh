#!/usr/bin/env bash
set -euo pipefail

root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-observe-7d07a421
base=f7cf0f6092b92e0ab2b813bc8acd8ee132a288e5
branch=codex/sz-current-pi0-dvac-observe
path=toolkits/probe_pi0_dvac_real_parity.py
subject='test: add real pi0 DVAC parity gate'

cd "$root"
head=$(git rev-parse HEAD)
test "$(git branch --show-current)" = "$branch"
test -z "$(git diff --name-only)"

if test "$head" = "$base"; then
  test "$(git diff --cached --name-only)" = "$path"
  test "$(git diff --cached --numstat)" = $'359\t0\t'"$path"
  git diff --cached --check
  author_name=$(git show -s --format=%an "$base")
  author_email=$(git show -s --format=%ae "$base")
  test -n "$author_name"
  test -n "$author_email"
  git -c user.name="$author_name" -c user.email="$author_email" \
    commit -m "$subject"
  head=$(git rev-parse HEAD)
else
  test "$(git rev-parse HEAD^)" = "$base"
  test "$(git show -s --format=%s HEAD)" = "$subject"
  test -z "$(git status --porcelain=v1)"
fi

test "$(git rev-parse HEAD^)" = "$base"
test "$(git show -s --format=%s HEAD)" = "$subject"
test "$(git show --format= --name-only HEAD | sed '/^$/d')" = "$path"
test -z "$(git status --porcelain=v1)"

GIT_CONFIG_NOSYSTEM=0 timeout --signal=TERM --kill-after=15s 120s \
  git push --porcelain personal "HEAD:refs/heads/$branch"

remote_sha=$(git ls-remote personal "refs/heads/$branch" | awk '{print $1}')
local_sha=$(git rev-parse HEAD)
upstream_sha=$(git rev-parse '@{upstream}')
test "$remote_sha" = "$local_sha"
test "$upstream_sha" = "$local_sha"
test "$(git rev-list --left-right --count '@{upstream}...HEAD')" = $'0\t0'
test -z "$(git status --porcelain=v1)"

echo "PI0_PARITY_COMMIT=$local_sha"
echo "PI0_PARITY_REMOTE=$remote_sha"
echo "PI0_PARITY_UPSTREAM=$upstream_sha"
echo 'PI0_PARITY_AHEAD_BEHIND=0 0'
echo 'PI0_PARITY_WORKTREE=CLEAN'
echo 'FORCE_USED=0'
echo 'GPU_RAY_MODEL_SIM_USED=0'
