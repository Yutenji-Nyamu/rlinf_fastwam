#!/usr/bin/env bash
set -u

RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-formal100-current-4gpu128train64eval-ppo-matched-v1
DRIVER_PID="$(cat "$RUN/driver.pid" 2>/dev/null || true)"
OBSERVER_PID="$(cat "$RUN/resource_observer.pid" 2>/dev/null || true)"

echo '=== IDENTITY_AND_RECORDED_PIDS ==='
date --iso-8601=seconds
id
printf 'run=%s\ndriver_pid=%s\nobserver_pid=%s\n' "$RUN" "$DRIVER_PID" "$OBSERVER_PID"
for name in driver observer; do
  if [ "$name" = driver ]; then pid="$DRIVER_PID"; else pid="$OBSERVER_PID"; fi
  if [ -n "$pid" ] && [ -d "/proc/$pid" ]; then
    printf '%s_alive=true ' "$name"
    ps -o user=,pid=,ppid=,pgid=,etimes=,rss=,stat=,comm=,args= -p "$pid"
  else
    printf '%s_alive=false\n' "$name"
  fi
done
if [ -s "$RUN/timeout.pid" ]; then
  TIMEOUT_PID="$(cat "$RUN/timeout.pid")"
  printf 'timeout_pid_file=%s\n' "$TIMEOUT_PID"
  ps -o user=,pid=,ppid=,pgid=,etimes=,rss=,stat=,comm=,args= -p "$TIMEOUT_PID" 2>/dev/null || true
else
  echo 'timeout_pid_file=absent'
fi
echo 'live_timeout_processes_for_user:'
ps -u "$(id -u)" -o user=,pid=,ppid=,pgid=,etimes=,rss=,stat=,comm=,args= \
  | awk '$8=="timeout" {print}' || true

echo '=== RAY_AND_GPU_NOW ==='
printf 'raylet='; pgrep -u "$(id -u)" -x raylet | tr '\n' ',' || true; echo
printf 'gcs_server='; pgrep -u "$(id -u)" -x gcs_server | tr '\n' ',' || true; echo
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true

echo '=== DRIVER_LOG_STAT_AND_TAIL150 ==='
stat -c 'bytes=%s mtime=%y path=%n' "$RUN/driver.log" 2>/dev/null || true
tail -n 150 "$RUN/driver.log" 2>/dev/null || true

echo '=== FIRST_FATAL_MATCHES ==='
grep -aEin -m 40 'Traceback|CUDA out of memory|OutOfMemoryError|WorkerCrashedError|SIGKILL|Killed process|NCCL.*(error|timeout)|RayTaskError|illegal instruction|segmentation fault|fatal|exception|error:' \
  "$RUN/driver.log" 2>/dev/null || true
first_trace="$(grep -aEin -m 1 'Traceback|CUDA out of memory|OutOfMemoryError|WorkerCrashedError|SIGKILL|Killed process|NCCL.*(error|timeout)|RayTaskError|illegal instruction|segmentation fault|fatal|exception|error:' "$RUN/driver.log" 2>/dev/null | cut -d: -f1 || true)"
if [ -n "$first_trace" ]; then
  start=$(( first_trace > 12 ? first_trace - 12 : 1 ))
  end=$(( first_trace + 100 ))
  printf 'first_fatal_context_lines=%s-%s\n' "$start" "$end"
  sed -n "${start},${end}p" "$RUN/driver.log"
else
  echo 'first_fatal_context=none'
fi

echo '=== LAST_COMPLETE_STEPS_AND_PROGRESS ==='
grep -a 'Global Step:' "$RUN/driver.log" 2>/dev/null | tail -n 5 || true
grep -aE 'Generating Rollout Epochs:|Evaluating|Saving|checkpoint|Success Rate|success rate' "$RUN/driver.log" 2>/dev/null | tail -n 50 || true

echo '=== CHECKPOINT_AND_EVAL_INVENTORY ==='
find "$RUN" -type d -name 'global_step_*' -printf '%T@ %p\n' 2>/dev/null | sort -n || true
printf 'eval_mp4_count='; find "$RUN/video/eval" -type f -name '*.mp4' 2>/dev/null | wc -l
find "$RUN/video/eval" -type f -name '*.mp4' -printf '%s %T@ %p\n' 2>/dev/null | sort -n | tail -n 20 || true
printf 'checkpoint_file_count='; find "$RUN" -path '*/global_step_*/*' -type f 2>/dev/null | wc -l

echo '=== OBSERVER_AND_RESOURCE_TAIL ==='
stat -c 'bytes=%s mtime=%y path=%n' "$RUN/resource.csv" "$RUN/resource_observer.log" 2>/dev/null || true
tail -n 20 "$RUN/resource.csv" 2>/dev/null || true
tail -n 80 "$RUN/resource_observer.log" 2>/dev/null || true

echo '=== EXIT_MARKER_INVENTORY ==='
find "$RUN" -maxdepth 2 -type f \
  \( -iname '*rc*' -o -iname '*exit*' -o -iname '*complete*' -o -iname '*timeout*' \) \
  -printf '%s %T@ %p\n' 2>/dev/null | sort -n || true
grep -aEin 'exit code|return code|driver.*rc|timeout|completed|finished|shutdown|SIGTERM|SIGINT|signal' \
  "$RUN/driver.log" "$RUN/resource_observer.log" 2>/dev/null | tail -n 80 || true

echo '=== CGROUP_MEMORY_EVENTS ==='
self_rel="$(awk -F: '$1=="0" {print $3}' /proc/self/cgroup)"
printf 'self_cgroup=%s\n' "$self_rel"
for cg in "/sys/fs/cgroup$self_rel" "/sys/fs/cgroup/user.slice/user-$(id -u).slice" /sys/fs/cgroup; do
  [ -d "$cg" ] || continue
  printf 'cgroup=%s\n' "$cg"
  for f in memory.current memory.peak memory.swap.current memory.swap.peak memory.events memory.events.local; do
    [ -r "$cg/$f" ] || continue
    printf '%s:\n' "$f"
    cat "$cg/$f"
  done
done

echo '=== RAY_SESSION_TERMINAL_LOGS ==='
if [ -e /tmp/ray/session_latest ]; then
  readlink -f /tmp/ray/session_latest
  for f in /tmp/ray/session_latest/logs/raylet.err /tmp/ray/session_latest/logs/gcs_server.err; do
    if [ -f "$f" ]; then
      stat -c 'bytes=%s mtime=%y path=%n' "$f"
      tail -n 80 "$f"
    fi
  done
  printf 'nonempty_worker_err_count='; find /tmp/ray/session_latest/logs -maxdepth 1 -type f -name 'worker-*.err' -size +0c 2>/dev/null | wc -l
  find /tmp/ray/session_latest/logs -maxdepth 1 -type f -name 'worker-*.err' -size +0c \
    -printf '%s %T@ %p\n' 2>/dev/null | sort -n | tail -n 20 || true
fi

echo 'SZ_GRPO_TERMINAL_DIAGNOSTIC_DONE'
