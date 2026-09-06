#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
cd "$WT"

echo '== status =='
git status --short

echo '== diff check =='
git diff --check
git diff --cached --check

echo '== py_compile =='
"$PY" -m py_compile \
  rlinf/algorithms/rlt/__init__.py \
  rlinf/algorithms/rlt/rollout.py \
  rlinf/algorithms/rlt/route.py \
  rlinf/algorithms/rlt/transition.py \
  rlinf/models/embodiment/openpi/__init__.py \
  rlinf/models/embodiment/openpi/openpi_action_model.py \
  rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py

echo '== ruff =='
if [[ -x /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/ruff ]]; then
  /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/ruff check \
    rlinf/algorithms/rlt/__init__.py \
    rlinf/algorithms/rlt/rollout.py \
    rlinf/algorithms/rlt/route.py \
    rlinf/algorithms/rlt/transition.py \
    rlinf/models/embodiment/openpi/__init__.py \
    rlinf/models/embodiment/openpi/openpi_action_model.py \
    rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py
else
  "$PY" -m ruff check \
    rlinf/algorithms/rlt/__init__.py \
    rlinf/algorithms/rlt/rollout.py \
    rlinf/algorithms/rlt/route.py \
    rlinf/algorithms/rlt/transition.py \
    rlinf/models/embodiment/openpi/__init__.py \
    rlinf/models/embodiment/openpi/openpi_action_model.py \
    rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py
fi

echo 'RLT_PRECHECK_OK'
