#!/usr/bin/env bash
set -euo pipefail

worktree=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
target=/root/autodl-tmp/datasets/robotwin2/source/9dc9299c163db059931898a9f0852098a61155a1/dataset/adjust_bottle/aloha-agilex_clean_50.zip

date -Is
nvidia-smi \
  --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader
free -h | sed -n '1,2p'
df -hT /root/autodl-tmp | sed -n '1,2p'
pgrep -af \
  'ray::|raylet|gcs_server|train_embodied_agent|train_sft|RoboTwin|robotwin|probe_robotwin_rlt' \
  || true

git -C "$worktree" status --short --branch
git -C "$worktree" rev-parse HEAD
git -C "$worktree" rev-list --left-right --count HEAD...@{upstream}

stat --printf='ZIP %n\nBYTES %s\nMTIME %y\n' "$target"
sha256sum "$target"
unzip -tqq "$target"
printf '%s\n' 'ZIP_TEST_OK'

for path in \
  /root/autodl-tmp/datasets/robotwin2/raw/9dc9299c163db059931898a9f0852098a61155a1/adjust_bottle \
  /root/autodl-tmp/datasets/robotwin2/intermediate/9dc9299c163db059931898a9f0852098a61155a1/adjust_bottle \
  /root/autodl-tmp/datasets/robotwin2/lerobot/pi0-aloha-clean50-v1
do
  if [[ -e "$path" ]]; then
    printf 'EXISTS %s\n' "$path"
  else
    printf 'MISSING %s\n' "$path"
  fi
done
