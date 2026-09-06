#!/usr/bin/env bash
set -u

RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-formal100-current-4gpu128train64eval-ppo-matched-v2
DRIVER=$(cat "$RUN/driver.pid" 2>/dev/null || true)
OBSERVER=$(cat "$RUN/resource_observer.pid" 2>/dev/null || true)

printf 'IDENTITY_TIME\n'
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
printf 'actor='; pgrep -u "$(id -u)" -fc '^ray::EmbodiedFSDPActor' || true
printf 'rollout='; pgrep -u "$(id -u)" -fc '^ray::MultiStepRolloutWorker' || true
printf 'env='; pgrep -u "$(id -u)" -fc '^ray::EnvWorker' || true
printf 'gcs='; pgrep -u "$(id -u)" -fc 'gcs_server' || true
printf 'raylet='; pgrep -u "$(id -u)" -fc 'raylet' || true

printf 'PROGRESS\n'
grep -aE 'Global Step:|Generating Rollout Epochs:' "$RUN/driver.log" | tail -n 18 || true

printf 'KEY_FILE_STATS\n'
stat -c '%n\t%s\t%y' \
  "$RUN/resolved.yaml" "$RUN/launch_manifest.txt" "$RUN/driver.log" \
  "$RUN/metrics.log" "$RUN/resource.csv" "$RUN/resource_observer.log" \
  "$RUN/tensorboard/config.yaml" 2>/dev/null || true
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

printf 'RESOURCE_SERIES\n'
printf 'rows='; wc -l < "$RUN/resource.csv"
head -n 3 "$RUN/resource.csv"
tail -n 5 "$RUN/resource.csv"

printf 'LIVE_RESOURCES\n'
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
printf 'host_mem_available_kib='; awk '/^MemAvailable:/ {print $2}' /proc/meminfo
if test -n "$DRIVER" && test -r "/proc/$DRIVER/cgroup"; then
  rel=$(awk -F: '$1=="0" {print $3}' "/proc/$DRIVER/cgroup")
  cg="/sys/fs/cgroup$rel"
  printf 'cgroup=%s\n' "$cg"
  for name in memory.current memory.peak memory.swap.current memory.swap.peak; do
    test -r "$cg/$name" && printf '%s=%s\n' "$name" "$(cat "$cg/$name")"
  done
  cat "$cg/memory.events"
fi
df -h / /data

printf 'HEALTH_AND_EXIT\n'
for file in driver.exit resource_observer.exit; do
  if test -f "$RUN/$file"; then printf '%s=' "$file"; cat "$RUN/$file"; else printf '%s=missing_running_expected\n' "$file"; fi
done
for file in "$RUN/metrics.log" "$RUN/driver.log" "$RUN/resource_observer.log"; do
  test -f "$file" || continue
  fatal=$(grep -aEic 'Traceback|CUDA out of memory|OutOfMemoryError|WorkerCrashedError|SIGKILL|Killed process|NCCL.*(error|timeout)|RayTaskError|illegal instruction|segmentation fault|Failed to connect to GCS|GCS may have been killed|No space left' "$file" || true)
  printf '%s fatal_pattern_count=%s\n' "$file" "$fatal"
done
nonfinite=$(grep -aEic '(^|[^A-Za-z])(nan|inf|-inf)([^A-Za-z]|$)' "$RUN/metrics.log" || true)
printf 'metrics_nonfinite_token_count=%s\n' "$nonfinite"
printf 'SZ_GRPO_V2_CURVE_ARTIFACT_AUDIT_OK\n'
