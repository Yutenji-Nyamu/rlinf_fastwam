#!/usr/bin/env bash
set -euo pipefail
for pid in 399688 399691; do
  echo "pid=$pid thread_count=$(find /proc/$pid/task -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)"
  ps -L -p "$pid" -o tid=,%cpu=,stat=,wchan:32=,comm= --sort=-%cpu | head -n 18
  echo 'wchan_counts:'
  ps -L -p "$pid" -o wchan= | sort | uniq -c | sort -nr | head -n 12
done
