#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin
python=/root/autodl-tmp/RLinf/.venv/bin/python

test "$(git -C "$repo" rev-parse HEAD)" = \
  6d0db56bf26f972cd27fa29535f5eb939e80e5bf
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
cd "$repo"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo"
export REPO_PATH="$repo"
export EMBODIED_PATH="$repo/examples/embodiment"

"$python" -m ruff check --no-cache \
  rlinf/models/embodiment/openpi/openpi_ogpo.py \
  rlinf/models/embodiment/openpi/openpi_action_model.py \
  tests/embodiment/test_openpi_ogpo_adapter.py
"$python" -m pytest -q -p no:cacheprovider \
  tests/embodiment/test_openpi_ogpo_adapter.py
git diff --check
