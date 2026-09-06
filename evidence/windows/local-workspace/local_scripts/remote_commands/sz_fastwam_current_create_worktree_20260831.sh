#!/usr/bin/env bash
set -euo pipefail

REPO=/data/chenyiteng/projects/rlinf-shenzhen/RLinf
TARGET=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
BRANCH=codex/sz-fastwam-current-rlinf-grpo
BASE=554c6dc8d586162d9444c01fa88308ed4f5203d0

test "$(git -C "${REPO}" cat-file -t "${BASE}")" = commit
test ! -e "${TARGET}"
if git -C "${REPO}" show-ref --verify --quiet "refs/heads/${BRANCH}"; then
  echo "branch already exists: ${BRANCH}" >&2
  exit 1
fi
mkdir -p "$(dirname "${TARGET}")"
git -C "${REPO}" worktree add -b "${BRANCH}" "${TARGET}" "${BASE}"
test "$(git -C "${TARGET}" rev-parse HEAD)" = "${BASE}"
test -z "$(git -C "${TARGET}" status --porcelain)"
printf 'worktree=%s\nbranch=%s\nhead=%s\n' \
  "${TARGET}" "$(git -C "${TARGET}" branch --show-current)" "$(git -C "${TARGET}" rev-parse HEAD)"
