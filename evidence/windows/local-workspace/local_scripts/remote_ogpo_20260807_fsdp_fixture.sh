#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
cd "$repo"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo"
export CUDA_VISIBLE_DEVICES=0,1

/root/autodl-tmp/RLinf/.venv/bin/torchrun \
  --standalone \
  --nproc_per_node=2 \
  tests/embodiment/ogpo_fsdp_ema_fixture.py
