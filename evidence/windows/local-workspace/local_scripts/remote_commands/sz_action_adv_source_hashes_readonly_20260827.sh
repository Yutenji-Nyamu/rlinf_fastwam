#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-dvac-action-adv
cd "$WT"
sha256sum \
  rlinf/algorithms/dvac_train_weighting.py \
  rlinf/workers/actor/embodied_fsdp_actor_worker.py \
  rlinf/algorithms/utils.py \
  examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi.yaml \
  tests/unit_tests/test_dvac_train_weighting.py
