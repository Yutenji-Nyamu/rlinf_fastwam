#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-dvac-action-adv
test "$(git -C "$WT" rev-parse HEAD)" = 0e28ac6f09f821ea12e7d54eba7118ce0000ca86
test "$(git -C "$WT" branch --show-current)" = codex/sz-grpo-dvac-action-adv
test -z "$(git -C "$WT" status --short)"
cd "$WT"
sha256sum \
  rlinf/workers/actor/embodied_fsdp_actor_worker.py \
  rlinf/algorithms/utils.py \
  examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi.yaml \
  tests/unit_tests/test_dvac_train_weighting.py
echo SZ_ACTION_ADV_UPLOAD_PREFLIGHT_OK
