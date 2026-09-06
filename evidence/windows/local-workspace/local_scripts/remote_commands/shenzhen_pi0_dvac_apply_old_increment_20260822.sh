#!/usr/bin/env bash
set -euo pipefail

wt=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-observe-7d07a421
cd "$wt"
test "$(git rev-parse HEAD)" = 7d07a4212ee6858cc333e1d4fab7a37256d1f839
test -z "$(git status --porcelain)"

set +e
git diff --binary 6d0db56bf26f972cd27fa29535f5eb939e80e5bf 61996e15cc7f5a32bd6012b61b20893d94636c82 -- \
  rlinf/models/embodiment/openpi/openpi_action_model.py \
  rlinf/workers/rollout/hf/huggingface_worker.py \
  rlinf/workers/env/env_worker.py \
  rlinf/utils/dvac_telemetry.py \
  evaluations/robotwin/robotwin_adjust_bottle_openpi_dvac_eval.yaml \
  tests/unit_tests/test_dvac_telemetry.py | git apply --3way
apply_rc=$?
set -e

echo "GIT_APPLY_RC=$apply_rc"
echo "STATUS_BEGIN"
git status --short
echo "STATUS_END"
echo "UNMERGED_BEGIN"
git diff --name-only --diff-filter=U
echo "UNMERGED_END"

test "$apply_rc" -ne 0
mapfile -t unmerged < <(git diff --name-only --diff-filter=U | sort)
test "${#unmerged[@]}" -eq 2
test "${unmerged[0]}" = rlinf/models/embodiment/openpi/openpi_action_model.py
test "${unmerged[1]}" = rlinf/workers/env/env_worker.py
test -f rlinf/utils/dvac_telemetry.py
test -f evaluations/robotwin/robotwin_adjust_bottle_openpi_dvac_eval.yaml
test -f tests/unit_tests/test_dvac_telemetry.py
