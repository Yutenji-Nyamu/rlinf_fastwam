#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
test "$(git -C "$repo" rev-parse HEAD)" = \
  6d0db56bf26f972cd27fa29535f5eb939e80e5bf

cd "$repo"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export REPO_PATH="$repo"
export EMBODIED_PATH="$repo/examples/embodiment"
export CUDA_VISIBLE_DEVICES=0

/root/autodl-tmp/RLinf/.venv/bin/python -
