#!/usr/bin/env bash
set -eu
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc
robotwin=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
date -Is
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits
free -h
ps -p 321933,322685,3176215,1568973 -o pid,etime,stat,comm
head -n 35 "$robotwin/robotwin/envs/vector_env.py"
sed -n '1,190p' "$root/rlinf/models/embodiment/openpi/dataconfig/robotwin_aloha_dataconfig.py"
git -C "$root" diff --check
git -C "$root" status --short
git -C "$root" diff --numstat
