#!/usr/bin/env bash
set -u

for pid in 475028 475036 475052 475065 475184 475379; do
  if test -r "/proc/${pid}/status"; then
    echo "[PID ${pid}]"
    grep -E '^(Name|State|Threads|VmRSS|voluntary_ctxt_switches|nonvoluntary_ctxt_switches):' "/proc/${pid}/status" || true
    printf 'task-wchan-counts: '
    for wchan_file in /proc/${pid}/task/*/wchan; do cat "${wchan_file}" 2>/dev/null || true; echo; done | sort | uniq -c | sort -nr | head -n 12
    echo "open-files:"
    ls -l "/proc/${pid}/fd" 2>/dev/null | grep -E '\.(so|py|pyc|json|lock|bin|safetensors|pt|pth|ckpt)|/tmp|/root/autodl-tmp' | tail -n 25 || true
  fi
done

echo "[TRACE_ONE_ACTOR]"
if command -v strace >/dev/null 2>&1; then
  timeout 4 strace -f -p 475028 -e trace=futex,openat,read,write,statx,newfstatat 2>&1 | tail -n 100 || true
else
  echo "strace unavailable"
fi
