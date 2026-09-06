#!/usr/bin/env bash
set -u
for wt in \
 /data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-pi0-robotwin-7d07a421 \
 /data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-dvac-action-adv-fix \
 /data/chenyiteng/projects/rlinf-shenzhen/worktrees/prism-dvac-rank-rloo \
 /data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-dvac-action-adv-fix \
 /data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-robotwin-rl; do
  printf 'WT=%s\n' "$wt"
  printf 'HEAD='; git -C "$wt" rev-parse HEAD
  printf 'BRANCH='; git -C "$wt" branch --show-current
  printf 'STATUS\n'; git -C "$wt" status --short --untracked-files=all | sed -n '1,8p'
  test -d "$wt/evidence/completed_runs_20260822_20260903" && du -sh "$wt/evidence/completed_runs_20260822_20260903" || true
done
