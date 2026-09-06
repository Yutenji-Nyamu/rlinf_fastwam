#!/usr/bin/env bash
set -euo pipefail

worktree=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
dsrl_worktree=/root/autodl-tmp/RLinf_fastwam_rlinf
zip=/root/autodl-tmp/datasets/robotwin2/source/9dc9299c163db059931898a9f0852098a61155a1/dataset/adjust_bottle/aloha-agilex_clean_50.zip

printf '%s\n' '=== identity_resources ==='
date -Is
hostname
id -u
nvidia-smi \
  --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader
free -h
df -hT /root/autodl-tmp

printf '%s\n' '=== relevant_processes ==='
pgrep -af \
  'ray::|raylet|gcs_server|train_embodied_agent|train_sft|RoboTwin|robotwin|rlt_stage1|torchrun' \
  || true

printf '%s\n' '=== git ==='
git -C "$worktree" status --short --branch
git -C "$worktree" rev-parse HEAD
git -C "$worktree" rev-list --left-right --count HEAD...@{upstream}
git -C "$dsrl_worktree" status --short --branch

printf '%s\n' '=== source_and_targets ==='
stat --printf='ZIP %n\nBYTES %s\nMTIME %y\n' "$zip"
sha256sum "$zip"
unzip -tqq "$zip"
printf '%s\n' 'ZIP_TEST_OK'

for path in \
  /root/autodl-tmp/datasets/robotwin2/raw/9dc9299c163db059931898a9f0852098a61155a1/adjust_bottle \
  /root/autodl-tmp/datasets/robotwin2/intermediate/9dc9299c163db059931898a9f0852098a61155a1/adjust_bottle \
  /root/autodl-tmp/datasets/robotwin2/canonical/pi0-aloha-clean50-contract-ep0-v1 \
  /root/autodl-tmp/datasets/robotwin2/canonical/pi0-aloha-clean50-v1 \
  "$worktree/logs/20260729_rlt_stage1_s1a_smoke_v1" \
  "$worktree/logs/20260729_rlt_stage1_s1b_batchfit_v1"
do
  if [[ -e "$path" ]]; then
    du -sh "$path"
  else
    printf 'MISSING %s\n' "$path"
  fi
done

printf '%s\n' '=== runtime_inputs ==='
readlink -f /root/autodl-tmp/RLinf/.venv/bin/python
/root/autodl-tmp/RLinf/.venv/bin/python -B -c \
  'import torch, datasets, lerobot; print("torch", torch.__version__); print("datasets", datasets.__version__); print("lerobot", getattr(lerobot, "__version__", "unknown"))'
find /root/autodl-tmp/checkpoints \
  -maxdepth 4 \
  -type f \
  -name norm_stats.json \
  -printf '%s|%TY-%Tm-%TdT%TH:%TM:%TS%Tz|%p\n' \
  2>/dev/null \
  | head -20
