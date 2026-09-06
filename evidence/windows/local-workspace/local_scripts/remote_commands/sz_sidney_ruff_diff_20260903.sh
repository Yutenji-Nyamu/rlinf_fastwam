#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
cd "$WT"
"$PY" -m ruff check --fix --diff rlinf/utils/ckpt_convertor/openpi/lerobot_pi05_to_openpi_rlinf.py
