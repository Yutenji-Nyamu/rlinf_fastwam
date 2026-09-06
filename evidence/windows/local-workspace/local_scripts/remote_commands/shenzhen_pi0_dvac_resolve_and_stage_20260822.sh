#!/usr/bin/env bash
set -euo pipefail

wt=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-observe-7d07a421
cd "$wt"
test "$(git rev-parse HEAD)" = 7d07a4212ee6858cc333e1d4fab7a37256d1f839

printf '%s  %s\n' \
  41407c5921d4c7be0fe17d27d4a68a445e389cb7bcd74e80cc623751594ef0db \
  rlinf/models/embodiment/openpi/openpi_action_model.py \
  2918fe82f6fccb93fa4a3cff4a080c26995ed7c3c31831884a1100925ceb3644 \
  rlinf/workers/env/env_worker.py \
  31fb3846667b360be6cc6149d0bd841ae6a30b6a08e3844449da16eb5298b8d4 \
  rlinf/workers/rollout/hf/huggingface_worker.py \
  ead027934a8f8725cf08aca41feb50d9133d3630915b03cd5449ab425c3cbde7 \
  evaluations/robotwin/robotwin_adjust_bottle_openpi_dvac_eval.yaml \
  24c778354316d54f32cf80700896d839314de863a38fc148f8f3b6e27e3d4905 \
  tests/unit_tests/test_dvac_telemetry.py | sha256sum --check

if grep -R -n -E '^(<<<<<<<|=======|>>>>>>>)' \
  rlinf/models/embodiment/openpi/openpi_action_model.py \
  rlinf/workers/env/env_worker.py \
  rlinf/workers/rollout/hf/huggingface_worker.py; then
  echo 'conflict marker remains' >&2
  exit 1
fi

grep -q 'return_dvac_telemetry: bool = False' \
  rlinf/models/embodiment/openpi/openpi_action_model.py
grep -q 'telemetry is not supported' \
  rlinf/models/embodiment/openpi/openpi_action_model.py
grep -q '"rtc_enabled"' rlinf/workers/rollout/hf/huggingface_worker.py
grep -q 'from rlinf.utils.dvac_telemetry import DVACEpisodeWriter' \
  rlinf/workers/env/env_worker.py
grep -q 'from rlinf.utils.env_helpers import HistoryManager, SmoothInterveneController' \
  rlinf/workers/env/env_worker.py
grep -q 'enabled: false' evaluations/robotwin/robotwin_adjust_bottle_openpi_dvac_eval.yaml
grep -q '7d07a4212ee6858cc333e1d4fab7a37256d1f839' \
  evaluations/robotwin/robotwin_adjust_bottle_openpi_dvac_eval.yaml

git add -- \
  rlinf/models/embodiment/openpi/openpi_action_model.py \
  rlinf/workers/rollout/hf/huggingface_worker.py \
  rlinf/workers/env/env_worker.py \
  rlinf/utils/dvac_telemetry.py \
  evaluations/robotwin/robotwin_adjust_bottle_openpi_dvac_eval.yaml \
  tests/unit_tests/test_dvac_telemetry.py

test -z "$(git diff --name-only --diff-filter=U)"
git diff --cached --check
mapfile -t changed < <(git diff --cached --name-only | sort)
test "${#changed[@]}" -eq 6

echo "STAGED_NAMES_BEGIN"
printf '%s\n' "${changed[@]}"
echo "STAGED_NAMES_END"
echo "STAGED_STAT_BEGIN"
git diff --cached --stat
echo "STAGED_STAT_END"
echo "STATUS_BEGIN"
git status --short
echo "STATUS_END"
