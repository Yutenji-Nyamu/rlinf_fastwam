#!/usr/bin/env bash
set -u

printf '=== ADMIN_IDENTITY_TIME ===\n'
date --iso-8601=seconds
hostname
id

printf '=== ALL_USER_HEAVY_PROCESSES ===\n'
ps -eo user=,pid=,ppid=,etimes=,rss=,%mem=,%cpu=,stat=,comm=,args= --sort=-rss | head -n 35
printf '%s\n' '-- cpu leaders --'
ps -eo user=,pid=,ppid=,etimes=,rss=,%mem=,%cpu=,stat=,comm=,args= --sort=-%cpu | head -n 25

printf '=== GPU_PROCESS_OWNERS ===\n'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sort -u); do
  test -n "$pid" && ps -o user=,pid=,ppid=,etimes=,rss=,%mem=,%cpu=,stat=,comm=,args= -p "$pid" 2>/dev/null || true
done

printf '=== USER_AGGREGATES ===\n'
ps -eo user=,rss=,%cpu= --no-headers | awk '{rss[$1]+=$2; cpu[$1]+=$3; n[$1]++} END {for (u in rss) printf "%s processes=%d rss_gib=%.3f cpu_sum=%.1f\n",u,n[u],rss[u]/1048576,cpu[u]}' | sort -k3,3nr

printf '=== HOST_HEALTH ===\n'
free -h
cat /proc/pressure/memory 2>/dev/null || true
df -h / /home /data
printf 'oom_or_kill_recent='; journalctl -k --since '2026-08-23 00:00:00' --no-pager 2>/dev/null | grep -Eic 'out of memory|oom-kill|Killed process' || true

printf 'SZ_FINAL_ADMIN_REFRESH_OK\n'
