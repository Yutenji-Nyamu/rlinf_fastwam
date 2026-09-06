#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv
cd "$repo"
export PYTHONPATH="$repo"

git branch --show-current
git rev-parse HEAD
git status --short
git diff --check

"$venv/bin/ruff" check \
  rlinf/data/qam_transition_replay.py \
  rlinf/workers/actor/fsdp_qam_policy_worker.py \
  tests/embodiment/test_robotwin_qam_contract.py \
  tests/workers/test_qam_worker_helpers.py
"$venv/bin/ruff" format --check \
  rlinf/data/qam_transition_replay.py \
  rlinf/workers/actor/fsdp_qam_policy_worker.py \
  tests/embodiment/test_robotwin_qam_contract.py \
  tests/workers/test_qam_worker_helpers.py
"$venv/bin/python" -m py_compile \
  rlinf/data/qam_transition_replay.py \
  rlinf/workers/actor/fsdp_qam_policy_worker.py \
  tests/embodiment/test_robotwin_qam_contract.py \
  tests/workers/test_qam_worker_helpers.py
"$venv/bin/python" -m pytest -q \
  tests/algorithms/qam/test_core.py \
  tests/algorithms/qam/test_official_fixture.py \
  tests/embodiment/test_qam_openpi_adapter.py \
  tests/embodiment/test_robotwin_qam_contract.py \
  tests/workers/test_qam_worker_helpers.py

pgrep -af 'train_embodied_agent|qam_|rlt_stage2|raylet|gcs_server' || true
nvidia-smi \
  --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
