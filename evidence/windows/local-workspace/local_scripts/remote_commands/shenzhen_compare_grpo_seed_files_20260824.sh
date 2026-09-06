#!/usr/bin/env bash
set -euo pipefail
OLD=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-pi0-robotwin-7d07a421/rlinf/envs/robotwin/seeds
NEW=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current/rlinf/envs/robotwin/seeds
for name in train_seeds.json eval_seeds.json; do
  if cmp -s "$OLD/$name" "$NEW/$name"; then
    echo "$name=identical"
  else
    echo "$name=DIFFERENT"
  fi
done
