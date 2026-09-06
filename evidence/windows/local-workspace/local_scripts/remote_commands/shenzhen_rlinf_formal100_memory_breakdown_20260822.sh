#!/usr/bin/env bash
set -u

echo '=== TIME ==='
date --iso-8601=seconds

echo '=== MEMINFO_GIB ==='
awk '
  /^(MemTotal|MemFree|MemAvailable|Buffers|Cached|SReclaimable|Shmem|AnonPages|Mapped|Slab|PageTables|Committed_AS):/ {
    printf "%s %.3f GiB\n", $1, $2/1024/1024
  }
' /proc/meminfo

echo '=== USER_RSS_AGGREGATE ==='
ps -u "$(id -u)" -o rss=,comm= | awk '
  {rss[$2]+=$1; n[$2]++}
  END {for (k in rss) printf "%.3f GiB\t%d\t%s\n",rss[k]/1024/1024,n[k],k}
' | sort -nr | head -n 25

echo '=== TOP_RSS_PROCESSES ==='
ps -u "$(id -u)" -o pid=,ppid=,rss=,%mem=,stat=,etime=,comm=,cmd= --sort=-rss | head -n 30

echo '=== USER_RSS_TOTAL ==='
ps -u "$(id -u)" -o rss= | awk '{s+=$1} END{printf "%.3f GiB\n",s/1024/1024}'

echo '=== SHM_AND_RAY ==='
df -h /dev/shm /tmp
du -sh /tmp/ray/session_2026-08-21_16-33-54_935915_1375834 2>/dev/null || true

echo 'FORMAL100_MEMORY_BREAKDOWN_DONE'
