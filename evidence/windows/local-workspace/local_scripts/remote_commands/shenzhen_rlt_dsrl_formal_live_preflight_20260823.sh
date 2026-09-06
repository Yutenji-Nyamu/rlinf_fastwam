#!/usr/bin/env bash
set -euo pipefail

printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
printf 'identity='; id
printf '%s\n' '--- gpu ---'
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
printf '%s\n' '--- host ---'
free -h
df -h / /home /data
printf '%s\n' '--- current-user training/control processes ---'
ps -u "$(id -u)" -o pid,ppid,etimes,rss,stat,comm,args --sort=-rss \
  | awk 'NR==1 || /train_embodied_agent|train_vla_sft|raylet|gcs_server|monitor\.py|formal|rlt|dsrl/' \
  | head -n 80
printf '%s\n' '--- repo locks ---'
for pair in \
  "rlt:/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421" \
  "dsrl:/data/chenyiteng/projects/rlinf-current-dsrl/RLinf-7d07-dsrl-robotwin"
do
  name="${pair%%:*}"
  path="${pair#*:}"
  printf '%s_head=' "$name"; git -C "$path" rev-parse HEAD
  printf '%s_branch=' "$name"; git -C "$path" branch --show-current
  printf '%s_dirty=' "$name"; git -C "$path" status --porcelain | wc -l
done
printf '%s\n' '--- recent formal-like output roots ---'
find /data/chenyiteng/results -maxdepth 4 -type d \
  \( -iname '*rlt*formal*' -o -iname '*dsrl*formal*' \) \
  -printf '%T@ %TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null \
  | sort -nr | head -n 30 || true
