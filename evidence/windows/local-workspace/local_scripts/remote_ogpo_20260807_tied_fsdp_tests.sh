#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin
python_bin=/root/autodl-tmp/RLinf/.venv/bin/python
test "$(git -C "$repo" rev-parse HEAD)" = \
  6d0db56bf26f972cd27fa29535f5eb939e80e5bf
test "$(git -C "$repo" branch --show-current)" = codex/ogpo-pi0-robotwin

cd "$repo"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo"
export REPO_PATH="$repo"
export EMBODIED_PATH="$repo/examples/embodiment"

"$python_bin" -m pytest -q -p no:cacheprovider \
  tests/embodiment/test_openpi_ogpo_adapter.py

test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
export CUDA_VISIBLE_DEVICES=0,1
/root/autodl-tmp/RLinf/.venv/bin/torchrun \
  --standalone \
  --nproc_per_node=2 \
  tests/embodiment/ogpo_real_fsdp_ema_probe.py
