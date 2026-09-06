#!/usr/bin/env bash
set -u

RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-formal100-current-4gpu128train64eval-ppo-matched-v2
DRIVER=$(cat "$RUN/driver.pid" 2>/dev/null || true)
OBSERVER=$(cat "$RUN/resource_observer.pid" 2>/dev/null || true)
printf 'time=%s driver=%s observer=%s\n' "$(date --iso-8601=seconds)" "$DRIVER" "$OBSERVER"
for pid in "$DRIVER" "$OBSERVER"; do
  if test -n "$pid" && kill -0 "$pid" 2>/dev/null; then printf 'pid=%s alive=1\n' "$pid"; else printf 'pid=%s alive=0\n' "$pid"; fi
done
grep -aE 'Global Step:|Generating Rollout Epochs:' "$RUN/driver.log" | tail -n 8 || true
printf 'LATEST_METRIC_TABLE\n'
tail -n 40 "$RUN/metrics.log" 2>/dev/null || true
printf 'actor='; pgrep -u "$(id -u)" -fc '^ray::EmbodiedFSDPActor' || true
printf 'rollout='; pgrep -u "$(id -u)" -fc '^ray::MultiStepRolloutWorker' || true
printf 'env='; pgrep -u "$(id -u)" -fc '^ray::EnvWorker' || true
printf 'gcs='; pgrep -u "$(id -u)" -xc gcs_server || true
printf 'raylet='; pgrep -u "$(id -u)" -xc raylet || true
fatal=$(grep -aEic 'Traceback|CUDA out of memory|OutOfMemoryError|WorkerCrashedError|SIGKILL|Killed process|NCCL.*(error|timeout)|RayTaskError|illegal instruction|segmentation fault|Failed to connect to GCS|GCS may have been killed|No space left' "$RUN/driver.log" || true)
nonfinite=$(grep -aEic '(^|[^A-Za-z])(nan|inf|-inf)([^A-Za-z]|$)' "$RUN/metrics.log" || true)
printf 'fatal=%s nonfinite=%s\n' "$fatal" "$nonfinite"
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
printf 'host_mem_available_kib='; awk '/^MemAvailable:/ {print $2}' /proc/meminfo
if test -n "$DRIVER" && test -r "/proc/$DRIVER/cgroup"; then
  rel=$(awk -F: '$1=="0" {print $3}' "/proc/$DRIVER/cgroup")
  cg="/sys/fs/cgroup$rel"
  for name in memory.current memory.swap.current; do test -r "$cg/$name" && printf '%s=%s\n' "$name" "$(cat "$cg/$name")"; done
  cat "$cg/memory.events"
fi
tail -n 1 "$RUN/resource.csv" 2>/dev/null || true
printf 'checkpoint_count='; find "$RUN" -type d -name 'global_step_*' 2>/dev/null | wc -l
printf 'eval_mp4='; find "$RUN/video/eval" -type f -name '*.mp4' 2>/dev/null | wc -l
printf 'SZ_GRPO_V2_FINAL_STATUS_OK\n'
