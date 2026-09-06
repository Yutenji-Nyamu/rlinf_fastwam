#!/usr/bin/env bash
set -euo pipefail

root=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711
base=fc652fb49cd32350eca15734b5c7124c0b8c2c02
branch=codex/sz-fastwam-dvac-observe
subject='test: add real Fast-WAM DVAC parity gate'
expected_names=$'experiments/robotwin/fastwam_real_query_parity.py\nsrc/fastwam/models/wan22/fastwam.py\ntests/test_fastwam_real_query_parity.py'

cd "$root"
head=$(git rev-parse HEAD)
test "$(git branch --show-current)" = "$branch"
test -z "$(git diff --name-only)"

if test "$head" = "$base"; then
  test "$(git diff --cached --name-only)" = "$expected_names"
  read -r added deleted < <(
    git diff --cached --numstat | awk '{a += $1; d += $2} END {print a, d}'
  )
  test "$added" = 384
  test "$deleted" = 1
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
fi

test "$(git rev-parse HEAD^)" = "$base"
test "$(git show -s --format=%s HEAD)" = "$subject"
test "$(git show --format= --name-only HEAD | sed '/^$/d')" = "$expected_names"
test -z "$(git status --porcelain=v1)"

echo "FASTWAM_PARITY_PARENT=$(git rev-parse HEAD^)"
echo "FASTWAM_PARITY_COMMIT=$(git rev-parse HEAD)"
echo 'FASTWAM_PARITY_WORKTREE=CLEAN'
echo 'PUSH_USED=0'
echo 'GPU_RAY_MODEL_SIM_USED=0'
