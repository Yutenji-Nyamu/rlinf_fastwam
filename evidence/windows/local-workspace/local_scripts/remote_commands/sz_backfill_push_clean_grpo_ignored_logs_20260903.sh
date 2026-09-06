#!/usr/bin/env bash
set -euo pipefail
wt=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-pi0-robotwin-7d07a421
branch=codex/sz-7d07a421-grpo-pi0-robotwin
out=evidence/completed_runs_20260822_20260903
test "$(git -C "$wt" rev-parse HEAD)" = bebd2f3acca691a397e4526e09ebf8c092045fb0
git -C "$wt" restore --staged -- "$out" 2>/dev/null || true
find "$wt/$out" -type d -name 'ray_logs*' -prune -exec rm -rf -- {} +
git -C "$wt" add -f -- "$out"
if ! git -C "$wt" diff --cached --quiet; then
  git -C "$wt" commit -m 'Include GRPO metrics and key logs'
  git -C "$wt" push personal "$branch"
fi
head=$(git -C "$wt" rev-parse HEAD)
remote=$(git -C "$wt" ls-remote personal "refs/heads/$branch" | awk '{print $1}')
test "$head" = "$remote"
test -z "$(git -C "$wt" status --porcelain --untracked-files=all)"
printf 'PUSHED_LOGS\t%s\t%s\t%s\n' "$branch" "$head" "$(git -C "$wt" ls-files "$out" | wc -l)"
