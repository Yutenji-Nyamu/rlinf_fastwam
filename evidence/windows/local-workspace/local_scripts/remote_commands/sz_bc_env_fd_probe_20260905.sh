#!/usr/bin/env bash
set -eu
id
date -Is
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=6
export PYTHONPATH=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc:/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
export ASSETS_PATH=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
unset LD_PRELOAD RLINF_SCENE_FENCE_LIBRARY
nvidia-smi -i 6 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
free -h
timeout --signal=TERM --kill-after=60s 1200s /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -u -
