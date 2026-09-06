#!/usr/bin/env bash
set -u

pids='391536 391538'
date --iso-8601=seconds
printf '%s\n' '--- current CPU samples ---'
if command -v pidstat >/dev/null 2>&1; then
  pidstat -p 391536,391538 1 3 || true
else
  top -b -d 1 -n 2 -p 391536,391538 | tail -n 30 || true
fi
printf '%s\n' '--- thread wait-channel counts ---'
for pid in $pids; do
  printf 'pid=%s\n' "$pid"
  ps -L -p "$pid" -o wchan= 2>/dev/null | sort | uniq -c | sort -nr | head -n 30 || true
  printf 'main_syscall='; cat "/proc/$pid/syscall" 2>/dev/null || true
  printf 'main_stack\n'; cat "/proc/$pid/stack" 2>/dev/null || true
done
printf '%s\n' '--- process sockets ---'
ss -tpn 2>/dev/null | grep -E 'pid=(391536|391538)' | head -n 100 || true
printf '%s\n' '--- hot threads ---'
top -H -b -d 1 -n 2 -p 391536,391538 2>/dev/null | tail -n 80 || true
