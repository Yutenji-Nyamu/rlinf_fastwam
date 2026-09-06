#!/usr/bin/env bash
set -euo pipefail

root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-observe-7d07a421
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
cd "$root"
test "$(git diff --cached --name-only | wc -l)" -eq 6
test -z "$(git diff --name-only)"

python_files=(
  rlinf/models/embodiment/openpi/openpi_action_model.py
  rlinf/utils/dvac_telemetry.py
  rlinf/workers/env/env_worker.py
  rlinf/workers/rollout/hf/huggingface_worker.py
  tests/unit_tests/test_dvac_telemetry.py
)
"$venv/bin/python" -m ruff format "${python_files[@]}"
git add -- "${python_files[@]}"
test "$(git diff --cached --name-only | wc -l)" -eq 6
test -z "$(git diff --name-only)"
git diff --cached --check
git diff --cached --stat
