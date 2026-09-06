#!/usr/bin/env bash
set -euo pipefail

worktree=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
dataset=/root/autodl-tmp/datasets/robotwin2/canonical/pi0-aloha-clean50-contract-ep0-v1
model=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle
stats=${model}/physical-intelligence/robotwin/norm_stats.json
probe=/root/autodl-tmp/tmp/rlt_loader_contract_20260729.py

export PYTHONPATH=${worktree}:/root/autodl-tmp/RoboTwin_RLinf
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=
export JAX_PLATFORMS=cpu
export HF_LEROBOT_HOME=/root/autodl-tmp/datasets/robotwin2/canonical
export ROBOTWIN_RLT_CLEAN50_PATH="$dataset"
export ROBOTWIN_PI0_BASE_PATH="$model"
export ROBOTWIN_PI0_NORM_STATS_PATH="$stats"

test -f "$dataset/meta/info.json"
test -f "$stats"
test -f "$probe"

for global_loader_batch in 2 32
do
  printf '=== global_loader_batch=%s ===\n' "$global_loader_batch"
  export LOADER_GLOBAL_BATCH="$global_loader_batch"
  /root/autodl-tmp/RLinf/.venv/bin/torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=2 \
    "$probe"
done
