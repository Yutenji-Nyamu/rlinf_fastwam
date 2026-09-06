#!/usr/bin/env bash
set -u

log=/data/chenyiteng/results/rlinf-current-dsrl/formal-dsrl-pi0-robotwin-2gpu4env-200c-20260824-v2/run/driver.log

date --iso-8601=seconds
stat -c 'log_size=%s mtime=%y' "$log" 2>/dev/null || true
printf '%s\n' '--- update-related tail ---'
tr '\r' '\n' < "$log" 2>/dev/null \
  | grep -aEi 'warm-up|planned_optimizer|optimizer|updates|update_step|critic|actor_loss|critic_loss|Global Step:|Traceback|ERROR|Exception|OOM' \
  | tail -n 100 || true
printf '%s\n' '--- raw tail ---'
tail -n 80 "$log" 2>/dev/null || true
printf '%s\n' '--- gpu6-7 ---'
nvidia-smi -i 6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits
printf '%s\n' '--- memory ---'
free -h | sed -n '1,2p'
