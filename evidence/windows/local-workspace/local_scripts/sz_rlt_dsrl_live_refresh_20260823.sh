#!/usr/bin/env bash
set -euo pipefail

echo '=== IDENTITY ==='
date --iso-8601=seconds
hostname
id

echo '=== GPU ==='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true

echo '=== MEMORY_DISK ==='
free -h
df -h / /home /data

echo '=== RL_PROCESSES ==='
ps -eo user:16,pid,ppid,rss,etimes,cmd --sort=-rss | grep -E 'raylet|gcs_server|train_embodied_agent|train\.py|RoboTwin|rlt|dsrl' | grep -v grep | head -40 || true

echo '=== WORKTREES ==='
for repo in \
  /data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421 \
  /data/chenyiteng/projects/rlinf-current-dsrl/RLinf-7d07-dsrl-robotwin
do
  echo "REPO=$repo"
  git -C "$repo" rev-parse HEAD
  git -C "$repo" branch --show-current
  git -C "$repo" status --short
done

echo '=== PATHS ==='
for path in \
  /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python \
  /data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support \
  /data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50 \
  /data/chenyiteng/datasets/robotwin2
do
  if [ -e "$path" ]; then
    echo "EXISTS $path"
  else
    echo "MISSING $path"
  fi
done

echo '=== PROXY ==='
for url in https://github.com https://huggingface.co; do
  code=$(curl --proxy http://127.0.0.1:7890 -L -sS -o /dev/null -w '%{http_code}' --connect-timeout 10 --max-time 20 "$url" || true)
  echo "PROXY_HTTP $url $code"
done
