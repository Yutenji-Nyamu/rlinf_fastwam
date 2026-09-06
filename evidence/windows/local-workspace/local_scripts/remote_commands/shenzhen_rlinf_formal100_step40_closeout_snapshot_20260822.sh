#!/usr/bin/env bash
set -u

RUN=/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-formal100-4gpu128train64eval-official-v1
PID="$(cat "$RUN/driver.pid")"

echo '=== TIME ==='
date --iso-8601=seconds
echo '=== DRIVER ==='
printf 'pid=%s alive=' "$PID"
if kill -0 "$PID" 2>/dev/null; then echo yes; else echo no; fi
echo '=== STEP40_ARTIFACTS ==='
find "$RUN" -maxdepth 3 \( -type f -o -type d \) \
  \( -name '*40*' -o -name 'global_step_*' \) -printf '%y %p\n' 2>/dev/null | sort | tail -n 30
echo '=== PROGRESS ==='
grep -aE 'Global Step|Generating Rollout Epochs:' "$RUN/driver.log" | tail -n 16
echo '=== EVAL_SAVE_TAIL ==='
grep -aEi 'eval|success|save|checkpoint|global_step_40' "$RUN/driver.log" | tail -n 30
echo '=== MEMORY ==='
CGROUP_REL="$(awk -F: '$1=="0" {print $3}' "/proc/$PID/cgroup")"
CGROUP="/sys/fs/cgroup$CGROUP_REL"
printf 'memory.current='; cat "$CGROUP/memory.current"
printf 'memory.swap.current='; cat "$CGROUP/memory.swap.current"
awk '/^(high|max|oom|oom_kill) / {print}' "$CGROUP/memory.events"
awk '/^(MemAvailable|AnonPages|Cached|Shmem):/ {print $1,$2}' /proc/meminfo
echo '=== GPU ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo '=== FATAL ==='
grep -aEi 'Traceback|CUDA out of memory|OutOfMemoryError|WorkerCrashedError|SIGKILL|Killed process' "$RUN/driver.log" | tail -n 10 || true
echo 'FORMAL100_STEP40_CLOSEOUT_SNAPSHOT_DONE'
