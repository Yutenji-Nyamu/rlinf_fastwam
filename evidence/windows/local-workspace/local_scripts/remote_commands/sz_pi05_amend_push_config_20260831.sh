#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-robotwin-rl
BRANCH=codex/sz-pi05-robotwin-rl
test "$(git -C "$WT" rev-parse HEAD)" = 8c420a9c40f1506521f3146b243dc2021b47a1ae
test "$(git -C "$WT" status --short | wc -l)" -eq 2
git -C "$WT" add -A
git -C "$WT" diff --cached --check
git -C "$WT" commit --amend --no-edit
test -z "$(git -C "$WT" status --porcelain)"
git -C "$WT" push -u personal "HEAD:refs/heads/$BRANCH"
head=$(git -C "$WT" rev-parse HEAD)
remote=$(git -C "$WT" ls-remote personal "refs/heads/$BRANCH" | awk '{print $1}')
test "$head" = "$remote"
printf 'head=%s\nremote=%s\nPI05_CONFIG_PUSHED\n' "$head" "$remote"
