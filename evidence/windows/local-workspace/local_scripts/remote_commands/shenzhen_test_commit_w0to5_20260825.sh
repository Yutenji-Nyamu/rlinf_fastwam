#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
cd "$WT"
"$PY" -m pytest -q tests/unit_tests/test_dvac_train_weighting.py
git diff --check
git add \
  rlinf/algorithms/dvac_train_weighting.py \
  rlinf/workers/actor/embodied_fsdp_actor_worker.py \
  examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi.yaml \
  tests/unit_tests/test_dvac_train_weighting.py
git commit -m 'feat(embodiment): support explicit DVAC weight endpoints'
git rev-parse HEAD
git status --short
echo SZ_W0TO5_TEST_COMMIT_OK
