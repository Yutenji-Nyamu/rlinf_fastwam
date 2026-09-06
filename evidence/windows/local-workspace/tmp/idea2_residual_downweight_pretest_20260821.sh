#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight
ROBOTWIN=/root/autodl-tmp/idea2_dvac_train_wamppo/RoboTwin_RLinf
PY=/root/autodl-tmp/RLinf/.venv/bin/python
OUT=/root/autodl-tmp/idea2_dvac_residual_downweight_pretest

mkdir -p "$OUT"
export REPO_PATH="$ROOT"
export EMBODIED_PATH="$ROBOTWIN"
export ROBOT_PLATFORM=ALOHA

"$PY" -m py_compile \
  "$ROOT/rlinf/algorithms/dvac_train_weighting.py" \
  "$ROOT/rlinf/workers/actor/fsdp_actor_worker.py" \
  "$ROOT/tests/unit_tests/test_dvac_train_weighting.py"

PYTHONPATH="$ROOT:$ROBOTWIN" "$PY" -m pytest -q \
  "$ROOT/tests/unit_tests/test_dvac_train_weighting.py" \
  "$ROOT/tests/unit_tests/test_dvac_telemetry.py"

PYTHONPATH="$ROOT:$ROBOTWIN" "$PY" \
  "$ROOT/examples/embodiment/train_embodied_agent.py" \
  --config-path "$ROOT/examples/embodiment/config" \
  --config-name robotwin_adjust_bottle_grpo_openpi_dvac_r_only_downweight_2step_smoke \
  --cfg job --resolve > "$OUT/resolved_r_only_downweight_2step.yaml"

PYTHONPATH="$ROOT:$ROBOTWIN" "$PY" \
  "$ROOT/examples/embodiment/train_embodied_agent.py" \
  --config-path "$ROOT/examples/embodiment/config" \
  --config-name robotwin_adjust_bottle_ppo_openpi \
  --cfg job --resolve > "$OUT/resolved_default_off_baseline.yaml"

PYTHONPATH="$ROOT" "$PY" -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  /root/autodl-tmp/idea2_residual_downweight_two_rank_probe_20260821.py

git -C "$ROOT" diff --check
git -C "$ROOT" status --short
echo PRETEST_OK
