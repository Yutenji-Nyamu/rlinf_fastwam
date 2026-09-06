#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711
cd "$WT"

echo "TIME_UTC=$(date -u +%FT%TZ)"
echo "HEAD=$(git rev-parse HEAD) BRANCH=$(git branch --show-current)"
echo "EVAL_FILES_BEGIN"
find third_party/RoboTwin experiments/robotwin -maxdepth 4 -type f \
  \( -name 'eval_policy.py' -o -name 'eval_robotwin_single.py' -o -name '_base_task.py' -o -name 'base_task.py' \) \
  -print | sort
echo "EVAL_FILES_END"

echo "RESET_AND_SUCCESS_CALLS_BEGIN"
grep -RInE --include='*.py' \
  'reset_model\(|model\.reset\(|success|is_success|take_action\(|get_obs\(' \
  experiments/robotwin/eval_robotwin_single.py \
  third_party/RoboTwin/script \
  third_party/RoboTwin/envs \
  third_party/RoboTwin/task_config 2>/dev/null | head -240 || true
echo "RESET_AND_SUCCESS_CALLS_END"

for f in \
  experiments/robotwin/eval_robotwin_single.py \
  third_party/RoboTwin/script/eval_policy.py \
  third_party/RoboTwin/envs/_base_task.py; do
  if test -f "$f"; then
    echo "FILE_BEGIN=$f"
    nl -ba "$f" | sed -n '1,320p'
    echo "FILE_END=$f"
  fi
done

echo FASTWAM_DVAC_EVALUATOR_CONTRACT_INSPECT_OK
