#!/usr/bin/env bash
set -euo pipefail

log=/root/autodl-tmp/tmp/rlt_clean50_download_20260729_v1.log
pidfile=/root/autodl-tmp/tmp/rlt_clean50_download_20260729_v1.pid
target=/root/autodl-tmp/datasets/robotwin2/source/9dc9299c163db059931898a9f0852098a61155a1/dataset/adjust_bottle/aloha-agilex_clean_50.zip

echo "STATUS_TIME $(date -Is)"
if [[ -f "$pidfile" ]]; then
  pid=$(cat "$pidfile")
  echo "pid=$pid"
  if kill -0 "$pid" 2>/dev/null; then
    ps -o pid=,ppid=,stat=,etime=,cmd= -p "$pid"
    echo "process_state=RUNNING"
  else
    echo "process_state=EXITED"
  fi
else
  echo "pidfile=MISSING"
fi

if [[ -e "$target" ]]; then
  stat --printf='target=%n\nbytes=%s\nmtime=%y\n' "$target"
else
  echo "target=MISSING"
fi

find "$(dirname "$target")" \
  -maxdepth 3 \
  -type f \
  -printf '%p|%s|%TY-%Tm-%TdT%TH:%TM:%TS%Tz\n' \
  2>/dev/null \
  | head -30 \
  || true

echo "=== log_tail ==="
tail -80 "$log" 2>/dev/null || true
