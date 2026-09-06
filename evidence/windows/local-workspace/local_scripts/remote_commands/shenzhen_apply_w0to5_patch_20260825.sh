#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current
BASE=66c863bc5a45e90cb5161b30af54355b1104c810
BRANCH=codex/sz-current-pi0-dvac-grpo-w0to5
PATCH=$(mktemp /tmp/sz-w0to5.XXXXXX.patch)
trap 'rm -f -- "$PATCH"' EXIT
cat >"$PATCH"

cd "$WT"
test "$(git rev-parse HEAD)" = "$BASE"
test -z "$(git status --porcelain)"
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  echo "branch already exists: $BRANCH" >&2
  exit 1
fi
git apply --check "$PATCH"
git switch -c "$BRANCH"
git apply "$PATCH"
git diff --check
git status --short
git diff --stat
echo SZ_W0TO5_PATCH_APPLIED_OK
