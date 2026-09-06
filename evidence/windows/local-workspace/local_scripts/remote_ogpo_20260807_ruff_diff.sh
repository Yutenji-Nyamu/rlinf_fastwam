#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin
python_bin=/root/autodl-tmp/RLinf/.venv/bin/python
cd "$repo"
"$python_bin" -m ruff check --no-cache --fix --diff \
  rlinf/data/ogpo_replay.py \
  rlinf/models/embodiment/openpi/openpi_action_model.py \
  rlinf/models/embodiment/openpi/openpi_ogpo.py
