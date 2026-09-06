#!/usr/bin/env bash
set -u

RUN=/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-formal100-4gpu128train64eval-official-v1
PID="$(cat "$RUN/driver.pid")"
echo '=== TIME ==='
date --iso-8601=seconds
echo '=== CGROUP_BYTES ==='
CGROUP_REL="$(awk -F: '$1=="0" {print $3}' "/proc/$PID/cgroup")"
CGROUP="/sys/fs/cgroup$CGROUP_REL"
printf 'driver_pid=%s alive=' "$PID"
if kill -0 "$PID" 2>/dev/null; then echo yes; else echo no; fi
printf 'memory.current='; cat "$CGROUP/memory.current"
printf 'memory.swap.current='; cat "$CGROUP/memory.swap.current"
awk '/^(high|max|oom|oom_kill) / {print}' "$CGROUP/memory.events"
echo '=== MEM_KIB ==='
awk '/^(MemAvailable|AnonPages|Cached|Shmem):/ {print $1,$2}' /proc/meminfo
echo '=== ENVWORKER_RSS_KIB ==='
ps -u "$(id -u)" -o pid=,rss=,etime=,comm=,cmd= | awk '$4=="ray::EnvWorker" {print $1,$2,$3,$4,$5}'
echo '=== USER_RSS_KIB ==='
ps -u "$(id -u)" -o rss= | awk '{s+=$1} END{print s}'
echo '=== LATEST_PROGRESS ==='
grep -aE 'Global Step|Generating Rollout Epochs:' "$RUN/driver.log" | tail -n 12
echo '=== CHECKPOINTS ==='
find "$RUN/checkpoints" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort -V | tail -n 5
echo '=== LATEST_METRIC_AND_FATAL ==='
tail -n 3 "$RUN/metrics.log" 2>/dev/null || true
printf 'fatal_count='
grep -aEi -c 'Traceback|CUDA out of memory|OutOfMemoryError|WorkerCrashedError|SIGKILL|Killed process' "$RUN/driver.log" || true
echo 'FORMAL100_MEMORY_TREND_SNAPSHOT_DONE'
