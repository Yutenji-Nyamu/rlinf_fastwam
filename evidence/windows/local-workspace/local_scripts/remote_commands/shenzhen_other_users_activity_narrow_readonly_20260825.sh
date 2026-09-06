#!/usr/bin/env bash
set -u
date --iso-8601=seconds
echo '=== gpu_compute_apps ==='
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
echo '=== relevant_active_processes ==='
ps -eo user:20,pid,ppid,etimes,%cpu,%mem,rss,stat,comm,args --sort=-rss \
  | awk '$1 ~ /^(liwenbo|zhangwei|toom|zhuanghuiping|guorenjie|qiufuwen)$/ {print}' \
  | head -n 60
echo '=== download_or_training_processes ==='
pgrep -af 'hf download|huggingface-cli|starwam|train|torchrun|ray::|RoboTwin|RLinf' 2>/dev/null | grep -v 'shenzhen_other_users_activity_narrow' | head -n 100 || true
