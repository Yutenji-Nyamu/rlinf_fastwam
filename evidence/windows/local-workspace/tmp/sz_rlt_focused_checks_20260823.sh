#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
PY="$VENV/bin/python"
RUFF="$VENV/bin/ruff"
cd "$WT"

TARGETS=(
  rlinf/algorithms/rlt/__init__.py
  rlinf/algorithms/rlt/rollout.py
  rlinf/algorithms/rlt/route.py
  rlinf/algorithms/rlt/transition.py
  rlinf/models/embodiment/openpi/__init__.py
  rlinf/models/embodiment/openpi/openpi_action_model.py
  rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py
  tests/unit_tests/test_robotwin_rlt_current_port.py
)

echo '== structural checks =='
git diff --check
git diff --cached --check
"$PY" -m py_compile "${TARGETS[@]}"
"$RUFF" check "${TARGETS[@]}"

echo '== hydra compose =='
"$PY" /tmp/check_rlt_current_compose.py

echo '== focused current-port fixture + official AR tests =='
"$PY" -m pytest -q \
  tests/unit_tests/test_robotwin_rlt_current_port.py \
  tests/unit_tests/test_rlt_token_transformer.py

echo 'RLT_FOCUSED_CHECKS_OK'
