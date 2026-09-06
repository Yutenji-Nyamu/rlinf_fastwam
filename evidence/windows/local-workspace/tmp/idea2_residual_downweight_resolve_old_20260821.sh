#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight
ROBOTWIN=/root/autodl-tmp/idea2_dvac_train_wamppo/RoboTwin_RLinf
PY=/root/autodl-tmp/RLinf/.venv/bin/python
OUT=/root/autodl-tmp/idea2_dvac_residual_downweight_pretest

export REPO_PATH="$ROOT"
export EMBODIED_PATH="$ROBOTWIN"
export ROBOT_PLATFORM=ALOHA

PYTHONPATH="$ROOT:$ROBOTWIN" "$PY" \
  "$ROOT/examples/embodiment/train_embodied_agent.py" \
  --config-path "$ROOT/examples/embodiment/config" \
  --config-name robotwin_adjust_bottle_grpo_openpi_dvac_train_2step_smoke \
  --cfg job --resolve > "$OUT/resolved_old_dvac_2step.yaml"

echo RESOLVE_OLD_OK
