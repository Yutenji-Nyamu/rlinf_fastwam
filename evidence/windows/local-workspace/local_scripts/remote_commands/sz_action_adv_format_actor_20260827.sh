#!/usr/bin/env bash
set -euo pipefail

worktree=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-dvac-action-adv
cd "$worktree"

source /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/activate
ruff format \
  rlinf/workers/actor/embodied_fsdp_actor_worker.py

git diff --check
git status --short
