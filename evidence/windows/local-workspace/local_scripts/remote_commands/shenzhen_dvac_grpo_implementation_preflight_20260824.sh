#!/usr/bin/env bash
set -euo pipefail

echo '=== identity_time ==='
date -Is
hostname
id
uptime

echo '=== rlt_process_and_progress ==='
ps -eo user,pid,ppid,pgid,stat,etimes,%cpu,%mem,rss,cmd --sort=-rss \
  | grep -E 'formal-current-ar-stage2-8env250-20260824-v4-warmup-fix|rlinf\.runner|ray::' \
  | head -40 || true
RLT_ROOT=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage2-8env250-20260824-v4-warmup-fix
if [[ -d "$RLT_ROOT" ]]; then
  find "$RLT_ROOT" -maxdepth 4 -type f \( -name '*.log' -o -name 'complete.json' -o -name 'events.out.tfevents*' \) \
    -printf '%T@ %s %p\n' | sort -nr | head -30
  grep -RhoE 'Global Step[^0-9]*[0-9]+|global_step[=/ _-]*[0-9]+|Step [0-9]+/[0-9]+' "$RLT_ROOT" \
    --include='*.log' 2>/dev/null | tail -30 || true
fi

echo '=== gpu ==='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true

echo '=== host_memory_storage ==='
free -h
awk '/MemAvailable|SwapFree|SwapTotal/ {print}' /proc/meminfo
df -hT / /home /data
df -i / /home /data

echo '=== project_worktrees ==='
for repo in \
  /data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-pi0-robotwin \
  /data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-observe \
  /data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar; do
  if [[ -e "$repo/.git" ]]; then
    echo "--- $repo"
    git -C "$repo" status --short --branch
    git -C "$repo" rev-parse HEAD
  fi
done

echo '=== lightweight_network ==='
set +u
source /etc/profile.d/mihomo-proxy.sh 2>/dev/null || true
set -u
curl -LIsS --max-time 12 https://github.com/ | head -n 1 || true
curl -LIsS --max-time 12 https://huggingface.co/ | head -n 1 || true

