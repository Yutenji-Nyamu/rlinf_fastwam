#!/usr/bin/env bash
set -uo pipefail

WT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711
cd "$WT" || exit 10
echo "TIME_UTC=$(date -u +%FT%TZ)"
echo "HEAD=$(git rev-parse HEAD)"
echo "BRANCH=$(git branch --show-current)"
echo "STAGED_BEGIN"
git diff --cached --name-status
echo "STAGED_END"
echo "STATUS_BEGIN"
git status --short --branch
echo "STATUS_END"
printf 'CONFIG_USER_NAME='; git config --get user.name || true
printf 'CONFIG_USER_EMAIL='; git config --get user.email || true
printf 'HEAD_AUTHOR_NAME='; git show -s --format=%an HEAD
printf 'HEAD_AUTHOR_EMAIL='; git show -s --format=%ae HEAD
echo FASTWAM_DVAC_COMMIT_IDENTITY_PROBE_OK
