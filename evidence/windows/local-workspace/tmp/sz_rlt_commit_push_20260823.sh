#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421
cd "$WT"

test "$(git rev-parse HEAD)" = 7d07a4212ee6858cc333e1d4fab7a37256d1f839
test "$(git branch --show-current)" = codex/sz-rlt-pi0-robotwin-ar
git diff --cached --check
git remote get-url personal

git commit -m "feat(rlt): port RoboTwin pi0 RLT to current RLinf"
git push -u personal codex/sz-rlt-pi0-robotwin-ar

echo '== final =='
echo "HEAD=$(git rev-parse HEAD)"
echo "BRANCH=$(git branch --show-current)"
git status --short
git log -1 --oneline --decorate
echo 'RLT_COMMIT_PUSH_OK'
