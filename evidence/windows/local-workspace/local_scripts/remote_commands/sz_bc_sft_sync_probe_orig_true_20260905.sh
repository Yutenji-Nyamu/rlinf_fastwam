#!/usr/bin/env bash
set -eu
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc
export REPO_PATH="$PYTHONPATH"
nvidia-smi -i 6 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
timeout --signal=TERM --kill-after=60s 1200s /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -u - --use-orig-params true
