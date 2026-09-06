#!/usr/bin/env bash
set -euo pipefail
for wt in \
  /data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-robotwin-rl \
  /data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current; do
  sha256sum "$wt/rlinf/envs/robotwin/seeds/train_seeds.json" "$wt/rlinf/envs/robotwin/seeds/eval_seeds.json"
done
