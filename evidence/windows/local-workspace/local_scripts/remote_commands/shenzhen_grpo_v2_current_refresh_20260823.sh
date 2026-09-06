#!/usr/bin/env bash
set -u

RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-formal100-current-4gpu128train64eval-ppo-matched-v2
UID_NOW=$(id -u)
DRIVER=$(cat "$RUN/driver.pid" 2>/dev/null || true)
OBSERVER=$(cat "$RUN/resource_observer.pid" 2>/dev/null || true)

printf 'IDENTITY_AND_TIME\n'
hostname
id
date --iso-8601=seconds
printf 'RUN=%s\nDRIVER=%s\nOBSERVER=%s\n' "$RUN" "$DRIVER" "$OBSERVER"

printf 'OWNED_PROCESS_STATE\n'
for pid in "$DRIVER" "$OBSERVER"; do
  if test -n "$pid" && kill -0 "$pid" 2>/dev/null; then
    ps -o pid=,ppid=,pgid=,etimes=,rss=,stat=,comm=,args= -p "$pid"
  else
    printf 'pid=%s alive=0\n' "$pid"
  fi
done
printf 'actor='; pgrep -u "$UID_NOW" -fc '^ray::EmbodiedFSDPActor' || true
printf 'rollout='; pgrep -u "$UID_NOW" -fc '^ray::MultiStepRolloutWorker' || true
printf 'env='; pgrep -u "$UID_NOW" -fc '^ray::EnvWorker' || true
printf 'gcs='; pgrep -u "$UID_NOW" -xc gcs_server || true
printf 'raylet='; pgrep -u "$UID_NOW" -xc raylet || true

printf 'PROGRESS_TAIL\n'
grep -aE 'Global Step:|Generating Rollout Epochs:' "$RUN/driver.log" 2>/dev/null | tail -n 30 || true
printf 'LATEST_METRICS\n'
tail -n 90 "$RUN/metrics.log" 2>/dev/null || true

printf 'KEY_FILE_STATS\n'
for file in resolved.yaml launch_manifest.txt driver.log metrics.log resource.csv resource_observer.log driver.exit resource_observer.exit; do
  if test -e "$RUN/$file"; then
    stat -c '%n\t%s\t%y' "$RUN/$file"
  else
    printf '%s\tMISSING\n' "$RUN/$file"
  fi
done
find "$RUN/tensorboard" -maxdepth 1 -type f -name 'events.out.tfevents.*' \
  -printf '%p\t%s\t%TY-%Tm-%Td %TH:%TM:%TS %Tz\n' 2>/dev/null | sort

printf 'ARTIFACT_INVENTORY\n'
du -sh "$RUN" 2>/dev/null || true
for kind in train eval; do
  count=$(find "$RUN/video/$kind" -type f -name '*.mp4' 2>/dev/null | wc -l)
  bytes=$(find "$RUN/video/$kind" -type f -name '*.mp4' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {printf "%.0f",s+0}')
  printf '%s_mp4_count=%s %s_mp4_bytes=%s\n' "$kind" "$count" "$kind" "$bytes"
done
printf 'checkpoint_count='; find "$RUN" -type d -name 'global_step_*' 2>/dev/null | wc -l
find "$RUN" -type d -name 'global_step_*' -print0 2>/dev/null | sort -zV | \
  while IFS= read -r -d '' path; do du -sh "$path"; done

printf 'RESOURCE_SERIES_BOUNDARY\n'
printf 'resource_rows='; wc -l < "$RUN/resource.csv" 2>/dev/null || true
head -n 2 "$RUN/resource.csv" 2>/dev/null || true
tail -n 5 "$RUN/resource.csv" 2>/dev/null || true

printf 'LIVE_GPU_MEMORY\n'
nvidia-smi -i 4,5,6,7 --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader,nounits

printf 'LIVE_HOST_AND_CGROUP_MEMORY\n'
free -h
printf 'host_mem_available_kib='; awk '/^MemAvailable:/ {print $2}' /proc/meminfo
CG_PID=''
if test -n "$DRIVER" && test -r "/proc/$DRIVER/cgroup"; then CG_PID=$DRIVER; fi
if test -z "$CG_PID" && test -n "$OBSERVER" && test -r "/proc/$OBSERVER/cgroup"; then CG_PID=$OBSERVER; fi
if test -n "$CG_PID"; then
  rel=$(awk -F: '$1=="0" {print $3}' "/proc/$CG_PID/cgroup")
  cg="/sys/fs/cgroup$rel"
  printf 'cgroup=%s\n' "$cg"
  for name in memory.current memory.peak memory.swap.current memory.swap.peak; do
    test -r "$cg/$name" && printf '%s=%s\n' "$name" "$(cat "$cg/$name")"
  done
  printf 'memory.events\n'; cat "$cg/memory.events" 2>/dev/null || true
  printf 'memory.stat.selected\n'
  awk '$1 ~ /^(anon|file|kernel|kernel_stack|pagetables|percpu|sock|shmem|file_mapped|file_dirty|inactive_anon|active_anon|inactive_file|active_file|slab|slab_reclaimable|slab_unreclaimable)$/ {print}' "$cg/memory.stat" 2>/dev/null || true
  printf 'memory.pressure\n'; cat "$cg/memory.pressure" 2>/dev/null || true
fi

printf 'OWNED_PROCESS_RSS_AGGREGATE_KIB\n'
ps -u "$UID_NOW" -o comm=,rss= | awk '
  /^ray::EnvWorker/ {env_n+=1; env+=$2}
  /^ray::MultiStep/ {roll_n+=1; roll+=$2}
  /^ray::Embodied/ {actor_n+=1; actor+=$2}
  /^raylet$/ {raylet_n+=1; raylet+=$2}
  /^gcs_server$/ {gcs_n+=1; gcs+=$2}
  END {
    printf "env_count=%d env_rss_kib=%.0f\n", env_n, env
    printf "rollout_count=%d rollout_rss_kib=%.0f\n", roll_n, roll
    printf "actor_count=%d actor_rss_kib=%.0f\n", actor_n, actor
    printf "raylet_count=%d raylet_rss_kib=%.0f\n", raylet_n, raylet
    printf "gcs_count=%d gcs_rss_kib=%.0f\n", gcs_n, gcs
  }'
ps -u "$UID_NOW" -o pid=,ppid=,rss=,etimes=,stat=,comm= --sort=-rss | head -n 20

printf 'HOST_ACTIVITY_AND_DISK\n'
vmstat 1 3
printf 'host_memory_pressure\n'; cat /proc/pressure/memory 2>/dev/null || true
df -h / /data

printf 'HEALTH_AND_EXIT\n'
for file in driver.exit resource_observer.exit; do
  if test -f "$RUN/$file"; then printf '%s=' "$file"; cat "$RUN/$file"; else printf '%s=missing\n' "$file"; fi
done
for file in "$RUN/metrics.log" "$RUN/driver.log" "$RUN/resource_observer.log"; do
  test -f "$file" || continue
  fatal=$(grep -aEic 'Traceback|CUDA out of memory|OutOfMemoryError|WorkerCrashedError|SIGKILL|Killed process|NCCL.*(error|timeout)|RayTaskError|illegal instruction|segmentation fault|Failed to connect to GCS|GCS may have been killed|No space left' "$file" || true)
  printf '%s fatal_pattern_count=%s\n' "$file" "$fatal"
done
nonfinite=$(grep -aEic '(^|[^A-Za-z])(nan|inf|-inf)([^A-Za-z]|$)' "$RUN/metrics.log" 2>/dev/null || true)
printf 'metrics_nonfinite_token_count=%s\n' "$nonfinite"
printf 'SZ_GRPO_V2_CURRENT_REFRESH_OK\n'

