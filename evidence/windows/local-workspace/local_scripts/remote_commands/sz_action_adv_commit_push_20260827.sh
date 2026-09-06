#!/usr/bin/env bash
set -euo pipefail

worktree=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-dvac-action-adv
base=0e28ac6f09f821ea12e7d54eba7118ce0000ca86
expected=$'examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi.yaml\nrlinf/algorithms/utils.py\nrlinf/workers/actor/embodied_fsdp_actor_worker.py\ntests/unit_tests/test_dvac_train_weighting.py'

cd "$worktree"
test "$(git rev-parse HEAD)" = "$base"
test "$(git branch --show-current)" = codex/sz-grpo-dvac-action-adv
test "$(git diff --name-only | sort)" = "$expected"
git diff --check

git add \
  examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi.yaml \
  rlinf/algorithms/utils.py \
  rlinf/workers/actor/embodied_fsdp_actor_worker.py \
  tests/unit_tests/test_dvac_train_weighting.py
git commit -m 'feat(embodiment): add DVAC action-level advantages'
git push personal HEAD:refs/heads/codex/sz-grpo-dvac-action-adv

git rev-parse HEAD
git status --short
git ls-remote personal refs/heads/codex/sz-grpo-dvac-action-adv
