#!/usr/bin/env bash
set -eu
rt=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-vector-render-lifecycle-fix-0008ae6
trial=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-oidn-toggle-20260904
test "$(git -C "$rt" rev-parse HEAD)" = 8c7380c118ce7ca8a4ea4df53d753adc8fab0df2
test -z "$(git -C "$rt" status --porcelain)"
test ! -e "$trial"
git -C "$rt" worktree add -b codex/sz-robotwin-oidn-toggle "$trial" 8c7380c118ce7ca8a4ea4df53d753adc8fab0df2
git -C "$trial" apply --check -
