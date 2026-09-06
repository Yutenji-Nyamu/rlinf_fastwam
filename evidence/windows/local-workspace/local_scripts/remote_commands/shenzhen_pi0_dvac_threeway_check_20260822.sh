#!/usr/bin/env bash
set -euo pipefail

wt=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-observe-7d07a421
cd "$wt"
test "$(git rev-parse HEAD)" = 7d07a4212ee6858cc333e1d4fab7a37256d1f839
test -z "$(git status --porcelain)"

echo "OLD_INCREMENT_STAT_BEGIN"
git diff --stat 6d0db56bf26f972cd27fa29535f5eb939e80e5bf 61996e15cc7f5a32bd6012b61b20893d94636c82 -- \
  rlinf/models/embodiment/openpi/openpi_action_model.py \
  rlinf/workers/rollout/hf/huggingface_worker.py \
  rlinf/workers/env/env_worker.py \
  rlinf/utils/dvac_telemetry.py \
  evaluations/robotwin/robotwin_adjust_bottle_openpi_dvac_eval.yaml \
  tests/unit_tests/test_dvac_telemetry.py
echo "OLD_INCREMENT_STAT_END"

echo "THREEWAY_CHECK_BEGIN"
git diff --binary 6d0db56bf26f972cd27fa29535f5eb939e80e5bf 61996e15cc7f5a32bd6012b61b20893d94636c82 -- \
  rlinf/models/embodiment/openpi/openpi_action_model.py \
  rlinf/workers/rollout/hf/huggingface_worker.py \
  rlinf/workers/env/env_worker.py \
  rlinf/utils/dvac_telemetry.py \
  evaluations/robotwin/robotwin_adjust_bottle_openpi_dvac_eval.yaml \
  tests/unit_tests/test_dvac_telemetry.py | git apply --3way --check --verbose
echo "THREEWAY_CHECK_END"

test -z "$(git status --porcelain)"
