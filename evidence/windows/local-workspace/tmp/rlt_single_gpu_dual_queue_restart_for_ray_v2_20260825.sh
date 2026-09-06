#!/usr/bin/env bash
set -euo pipefail
package_root=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dual480_20260825_runtime_package
queue_runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dual480_20260825_queue
old_queue_pid=$(cat "$queue_runtime/queue.pid")

bash -n "$package_root/queue_after_old.sh"
bash -n "$package_root/start_ray_head.sh"
kill -INT -- "-${old_queue_pid}"
for _ in $(seq 1 30); do
  kill -0 "$old_queue_pid" 2>/dev/null || break
  sleep 1
done
if kill -0 "$old_queue_pid" 2>/dev/null; then
  echo "old queue still alive: $old_queue_pid" >&2
  exit 2
fi
mv "$queue_runtime/queue.pid" "$queue_runtime/queue_v1.pid"
mv "$queue_runtime/queue.log" "$queue_runtime/queue_v1.log"

setsid bash "$package_root/queue_after_old.sh" >"$queue_runtime/queue.log" 2>&1 < /dev/null &
new_pid=$!
printf '%s\n' "$new_pid" >"$queue_runtime/queue.pid"
sleep 1

echo "old_queue_pid=$old_queue_pid"
echo "new_queue_pid=$new_pid"
ps -o pid,ppid,pgid,lstart,cmd -p "$new_pid"
ps -o pid,ppid,pgid,lstart,cmd -p 106844
sha256sum "$package_root"/*.sh
