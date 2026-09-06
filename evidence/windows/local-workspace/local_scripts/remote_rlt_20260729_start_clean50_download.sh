#!/usr/bin/env bash
set -euo pipefail

script=/root/autodl-tmp/tmp/rlt_clean50_download_20260729_v1.sh
log=/root/autodl-tmp/tmp/rlt_clean50_download_20260729_v1.log
pidfile=/root/autodl-tmp/tmp/rlt_clean50_download_20260729_v1.pid

if [[ ! -f "$script" ]]; then
  echo "FAIL: missing uploaded script $script" >&2
  exit 30
fi
if [[ -e "$log" || -e "$pidfile" ]]; then
  echo "FAIL: evidence path already exists; inspect before retry" >&2
  ls -l "$log" "$pidfile" 2>/dev/null || true
  exit 31
fi

chmod 700 "$script"
nohup bash "$script" >"$log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$pidfile"

sleep 1
if kill -0 "$pid" 2>/dev/null; then
  echo "STARTED pid=$pid"
else
  echo "PROCESS_EXITED_EARLY pid=$pid"
fi
echo "script=$script"
echo "log=$log"
echo "pidfile=$pidfile"
tail -40 "$log" || true
