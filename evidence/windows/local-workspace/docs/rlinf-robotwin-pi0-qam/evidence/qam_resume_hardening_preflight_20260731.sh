#!/usr/bin/env bash
set -euo pipefail

hostname
pwd
id -u
date '+%F %T %Z'

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
git -C "$repo" branch --show-current
git -C "$repo" rev-parse HEAD
git -C "$repo" status --short
git -C "$repo" rev-list --left-right --count '@{upstream}...HEAD'

ps -eo pid,etimes,%cpu,%mem,rss,args --sort=-rss |
  grep -E 'rlt_stage2|qam_|ray::|train_embodied_agent' |
  grep -v grep |
  head -20 || true
nvidia-smi \
  --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu \
  --format=csv,noheader
free -h
df -h /root/autodl-tmp
