#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export MPLBACKEND=Agg
/root/autodl-tmp/RLinf/.venv/bin/python -B /tmp/extract_rlt_step475_episode_probe_light_20260828.py
