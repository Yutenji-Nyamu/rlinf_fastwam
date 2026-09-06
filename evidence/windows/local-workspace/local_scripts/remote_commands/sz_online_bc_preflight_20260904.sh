#!/usr/bin/env bash
set -eu
export PYTHONDONTWRITEBYTECODE=1
date -Is
hostname
id
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader,nounits
free -h
df -h / /home /data
cat /proc/pressure/memory /proc/pressure/io
ps -eo user,pid,pgid,etime,rss,args | grep -E '[t]rain_embodied_agent|[g]cs_server|[r]aylet' | cut -c 1-650
find /data/chenyiteng/projects/rlinf-shenzhen -maxdepth 2 -type d -name .git -printf '%h\n'
find /home/chenyiteng/venvs -maxdepth 1 -mindepth 1 -type d -printf '%f\n'
for repo in /data/chenyiteng/projects/rlinf-shenzhen/RLinf /data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-robotwin-rl /data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo; do
  if test -d "$repo"; then
    printf 'REPO %s\n' "$repo"
    git --no-optional-locks -C "$repo" rev-parse HEAD
    git --no-optional-locks -C "$repo" status --short
    git -C "$repo" remote -v
  fi
done
