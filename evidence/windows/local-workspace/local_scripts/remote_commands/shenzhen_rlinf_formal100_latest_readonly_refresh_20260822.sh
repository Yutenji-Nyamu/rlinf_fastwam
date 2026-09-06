#!/usr/bin/env bash
set -u

RUN=/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-formal100-4gpu128train64eval-official-v1
PID="$(cat "$RUN/driver.pid")"

echo '=== TIME ==='
date --iso-8601=seconds
echo '=== IDENTITY ==='
id
echo '=== DRIVER ==='
printf 'pid=%s alive=' "$PID"
if kill -0 "$PID" 2>/dev/null; then echo yes; else echo no; fi
tr '\0' ' ' < "/proc/$PID/cmdline" 2>/dev/null || true
echo

echo '=== PROGRESS ==='
grep -aE 'Global Step:|Generating Rollout Epochs:' "$RUN/driver.log" | tail -n 24
echo '=== LATEST_COMPLETE_TABLE ==='
grep -a 'Global Step:' "$RUN/driver.log" | tail -n 1
echo '=== FATAL_COUNT ==='
grep -aEic 'Traceback|CUDA out of memory|OutOfMemoryError|WorkerCrashedError|SIGKILL|Killed process|NCCL.*(error|timeout)' "$RUN/driver.log" || true

echo '=== MEMORY ==='
CGROUP_REL="$(awk -F: '$1=="0" {print $3}' "/proc/$PID/cgroup")"
CGROUP="/sys/fs/cgroup$CGROUP_REL"
printf 'cgroup_path=%s\n' "$CGROUP"
printf 'memory.current='; cat "$CGROUP/memory.current"
printf 'memory.swap.current='; cat "$CGROUP/memory.swap.current"
awk '/^(low|high|max|oom|oom_kill) / {print}' "$CGROUP/memory.events"
awk '/^(MemTotal|MemAvailable|AnonPages|Cached|Shmem|SwapTotal|SwapFree):/ {print $1,$2}' /proc/meminfo
echo '=== ENVWORKER_RSS_KIB ==='
ps -u chenyiteng -o pid=,rss=,comm=,args= | awk '/ray::EnvWorker/ {sum+=$2; print} END {printf "sum_kib=%d\n",sum}'

echo '=== GPU ==='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits
echo '=== GPU_COMPUTE_APPS ==='
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true

echo '=== RUN_ARTIFACTS ==='
du -sb "$RUN" 2>/dev/null || true
printf 'metrics_log='; stat -c '%s %y %n' "$RUN/metrics.log" 2>/dev/null || true
printf 'driver_log='; stat -c '%s %y %n' "$RUN/driver.log" 2>/dev/null || true
printf 'tensorboard_files='; find "$RUN" -type f \( -name 'events.out.tfevents*' -o -name '*.tfevents' \) -printf '%s %T@ %p\n' 2>/dev/null | sort -n
echo 'checkpoints:'
for d in "$RUN"/robotwin_ppo_openpi/checkpoints/global_step_*; do
  [ -d "$d" ] || continue
  du -sb "$d"
  find "$d" -type f -name full_weights.pt -printf 'full_weights %s %p\n'
done
printf 'train_mp4_count='; find "$RUN/video/train" -type f -name '*.mp4' 2>/dev/null | wc -l
printf 'train_mp4_bytes='; find "$RUN/video/train" -type f -name '*.mp4' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {printf "%d\n",s}'
printf 'eval_mp4_count='; find "$RUN/video/eval" -type f -name '*.mp4' 2>/dev/null | wc -l
printf 'eval_mp4_bytes='; find "$RUN/video/eval" -type f -name '*.mp4' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {printf "%d\n",s}'
echo 'eval video directories:'
find "$RUN/video/eval" -mindepth 1 -maxdepth 2 -type d -printf '%p\n' 2>/dev/null | sort

echo '=== RESOURCE_FILES_SEARCH ==='
find "$RUN" "$(dirname "$RUN")" -maxdepth 3 -type f \( -iname '*resource*.csv' -o -iname '*resource*.log' -o -iname '*monitor*.csv' -o -iname '*monitor*.log' \) -printf '%s %T@ %p\n' 2>/dev/null | sort -u
echo 'FORMAL100_LATEST_READONLY_REFRESH_DONE'
