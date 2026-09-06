#!/usr/bin/env bash
set -euo pipefail

WORKTREE=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
DATASET=/root/autodl-tmp/datasets/robotwin2/canonical/pi0-aloha-clean50-v1
MODEL=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle
STATS=${MODEL}/physical-intelligence/robotwin/norm_stats.json
MANIFEST=/root/autodl-tmp/datasets/robotwin2/manifests/pi0-aloha-clean50-v1.json
PROBE=/root/autodl-tmp/tmp/rlt_loader_full_clean50_20260729.py

export PYTHONPATH=${WORKTREE}:/root/autodl-tmp/RoboTwin_RLinf
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=
export JAX_PLATFORMS=cpu
export HF_LEROBOT_HOME=/root/autodl-tmp/datasets/robotwin2/canonical
export ROBOTWIN_RLT_CLEAN50_PATH="$DATASET"
export ROBOTWIN_PI0_BASE_PATH="$MODEL"
export ROBOTWIN_PI0_NORM_STATS_PATH="$STATS"
export LOADER_GLOBAL_BATCH=32

test -f "$DATASET/meta/info.json"
test -f "$MANIFEST"
test -f "$STATS"
test -f "$PROBE"
printf 'START\t%s\n' "$(date --iso-8601=seconds)"
sha256sum "$MANIFEST" "$STATS"
/root/autodl-tmp/RLinf/.venv/bin/torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=2 \
  "$PROBE"
printf 'SUCCESS\t%s\n' "$(date --iso-8601=seconds)"
