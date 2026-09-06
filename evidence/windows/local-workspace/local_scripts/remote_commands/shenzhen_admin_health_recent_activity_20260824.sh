#!/usr/bin/env bash
set -euo pipefail

sudo -S -p '' bash -c '
set -euo pipefail
echo "=== admin_identity_time ==="
date -Is
hostname
id
uptime

echo "=== sessions_and_recent_logins ==="
w -h || true
last -F -n 30 | head -30 || true

echo "=== top_processes ==="
ps -eo user,pid,ppid,stat,etimes,%cpu,%mem,rss,cmd --sort=-rss | head -35

echo "=== gpu_memory_storage ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
free -h
df -hT / /home /data
df -i / /home /data

echo "=== top_level_recent_activity ==="
find /home /data -xdev -mindepth 1 -maxdepth 2 -mmin -4320 \
  -printf "%TY-%Tm-%TdT%TH:%TM %u %y %s %p\n" 2>/dev/null \
  | sort -r | head -120 || true

echo "=== service_and_kernel_health ==="
systemctl --failed --no-pager || true
journalctl -k --since "3 days ago" --no-pager \
  | grep -Ei "oom|out of memory|nvrm: xid|nvme.*error|i/o error|ext4.*error|xfs.*error" \
  | tail -80 || true
systemctl is-active mihomo || true
systemctl show mihomo -p ActiveEnterTimestamp -p NRestarts --no-pager || true
'
