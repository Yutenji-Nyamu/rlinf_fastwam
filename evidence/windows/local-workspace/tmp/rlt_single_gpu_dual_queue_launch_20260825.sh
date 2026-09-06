#!/usr/bin/env bash
set -euo pipefail
package_root=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dual480_20260825_runtime_package
queue_runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dual480_20260825_queue
mkdir -p "$queue_runtime"
if [ -e "$queue_runtime/queue.pid" ]; then
  echo 'queue.pid already exists' >&2
  exit 2
fi
setsid bash "$package_root/queue_after_old.sh" >"$queue_runtime/queue.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" >"$queue_runtime/queue.pid"
sleep 1
ps -o pid,ppid,pgid,lstart,cmd -p "$pid"
cat "$queue_runtime/queue_started_at.txt"
