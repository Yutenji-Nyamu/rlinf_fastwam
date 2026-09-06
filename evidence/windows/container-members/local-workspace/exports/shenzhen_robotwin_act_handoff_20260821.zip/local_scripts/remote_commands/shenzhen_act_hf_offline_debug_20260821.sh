#!/usr/bin/env bash
set -euo pipefail

source /home/chenyiteng/miniforge3/etc/profile.d/conda.sh
conda activate act

ROBOTWIN=/data/chenyiteng/projects/robotwin-native/RoboTwin
ACT_DIR="$ROBOTWIN/XPolicyLab/policy/ACT"
HF_LEAF=/data/chenyiteng/models/robotwin2-hf-a967b852/act_ckpt/act-adjust_bottle/demo_clean-50
RUN=/data/chenyiteng/runs/robotwin-native/adjust_bottle/three_stage_function_smoke_20260821_v1

test -s "$HF_LEAF/policy_last.ckpt"
test -s "$HF_LEAF/dataset_stats.pkl"
cd "$ACT_DIR"

EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1 \
timeout --signal=INT --kill-after=60s 1800s \
  bash eval.sh RoboTwin adjust_bottle "$HF_LEAF" aloha_agilex joint 0 0 0 act RoboTwin 2>&1 | \
  tee "$RUN/07_act_hf_offline_debug.log"

grep -F '[MAIN] eval finished' "$RUN/07_act_hf_offline_debug.log"
