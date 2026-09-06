#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin
test "$(git -C "$repo" rev-parse HEAD)" = \
  6d0db56bf26f972cd27fa29535f5eb939e80e5bf
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
cd "$repo"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo"
export REPO_PATH="$repo"
export EMBODIED_PATH="$repo/examples/embodiment"
export CUDA_VISIBLE_DEVICES=0,1

/root/autodl-tmp/RLinf/.venv/bin/python -m ruff check --no-cache \
  tests/embodiment/ogpo_real_fsdp_update_probe.py

/root/autodl-tmp/RLinf/.venv/bin/torchrun \
  --standalone \
  --nproc_per_node=2 \
  tests/embodiment/ogpo_real_fsdp_update_probe.py

printf 'POST_FULL_UPDATE_GPU_PROCESSES\n'
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits || true
