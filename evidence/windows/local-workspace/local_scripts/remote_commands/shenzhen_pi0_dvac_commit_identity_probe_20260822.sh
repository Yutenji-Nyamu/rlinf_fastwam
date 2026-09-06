#!/usr/bin/env bash
set -euo pipefail

root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-observe-7d07a421
cd "$root"
echo "HEAD=$(git rev-parse HEAD)"
echo "BRANCH=$(git branch --show-current)"
echo "LOCAL_USER_NAME=$(git config --get user.name || true)"
echo "LOCAL_USER_EMAIL=$(git config --get user.email || true)"
echo "OLD_TELEMETRY_AUTHOR=$(git show -s --format='%an <%ae>' 61996e15cc7f5a32bd6012b61b20893d94636c82)"
echo "OLD_TELEMETRY_COMMITTER=$(git show -s --format='%cn <%ce>' 61996e15cc7f5a32bd6012b61b20893d94636c82)"
echo "STATUS_BEGIN"
git status --short
echo "STATUS_END"
