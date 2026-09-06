#!/usr/bin/env bash
set -u

RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-formal100-current-4gpu128train64eval-ppo-matched-v2
DRIVER=$(cat "$RUN/driver.pid" 2>/dev/null || true)
OBSERVER=$(cat "$RUN/resource_observer.pid" 2>/dev/null || true)

printf 'TIME=%s\nRUN=%s\nDRIVER=%s\nOBSERVER=%s\n' "$(date --iso-8601=seconds)" "$RUN" "$DRIVER" "$OBSERVER"
for pid in "$DRIVER" "$OBSERVER"; do
  if test -n "$pid" && kill -0 "$pid" 2>/dev/null; then
    ps -o pid=,ppid=,pgid=,etimes=,rss=,stat=,comm=,args= -p "$pid"
  else
    printf 'PID=%s ALIVE=0\n' "$pid"
  fi
done

printf '%s\n' 'RAY_COUNTS_BEGIN'
ps -u "$(id -u)" -o pid=,rss=,args= | awk '
  /ray::EmbodiedFSDPActor/ {actor++; actor_rss+=$2}
  /ray::MultiStepRolloutWorker/ {rollout++; rollout_rss+=$2}
  /ray::EnvWorker/ {env++; env_rss+=$2}
  /gcs_server/ {gcs++; gcs_rss+=$2}
  /raylet/ {raylet++; raylet_rss+=$2}
  END {
    printf "ACTOR=%d ACTOR_RSS_KIB=%d\n", actor, actor_rss;
    printf "ROLLOUT=%d ROLLOUT_RSS_KIB=%d\n", rollout, rollout_rss;
    printf "ENV=%d ENV_RSS_KIB=%d\n", env, env_rss;
    printf "GCS=%d GCS_RSS_KIB=%d\n", gcs, gcs_rss;
    printf "RAYLET=%d RAYLET_RSS_KIB=%d\n", raylet, raylet_rss;
  }'
printf '%s\n' 'RAY_COUNTS_END'

printf '%s\n' 'PROGRESS_BEGIN'
grep -aE 'Global Step:|Generating Rollout Epochs:' "$RUN/driver.log" | tail -n 20 || true
printf '%s\n' 'PROGRESS_END'
printf '%s\n' 'LATEST_GLOBAL_TABLES_BEGIN'
grep -a 'Global Step:' "$RUN/driver.log" | tail -n 5 || true
printf '%s\n' 'LATEST_GLOBAL_TABLES_END'
printf '%s\n' 'LATEST_METRIC_CONTEXT_BEGIN'
grep -aEi 'advantage|policy_loss|actor_loss|approx_kl|clip_fraction|grad_norm|ratio_abs|success_once|return' "$RUN/driver.log" | tail -n 100 || true
printf '%s\n' 'LATEST_METRIC_CONTEXT_END'

fatal=$(grep -aEic 'Traceback|CUDA out of memory|OutOfMemoryError|WorkerCrashedError|SIGKILL|Killed process|NCCL.*(error|timeout)|RayTaskError|illegal instruction|segmentation fault|Failed to connect to GCS|GCS may have been killed' "$RUN/driver.log" || true)
nonfinite=$(grep -aEic '(^|[^A-Za-z])(nan|inf|-inf)([^A-Za-z]|$)' "$RUN/metrics.log" || true)
printf 'FATAL_COUNT=%s\nMETRICS_NONFINITE_TOKEN_COUNT=%s\n' "$fatal" "$nonfinite"

printf '%s\n' 'GPU4_7_BEGIN'
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
nvidia-smi -i 4,5,6,7 --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
printf '%s\n' 'GPU4_7_END'

if test -n "$DRIVER" && test -r "/proc/$DRIVER/cgroup"; then
  rel=$(awk -F: '$1=="0" {print $3}' "/proc/$DRIVER/cgroup")
  cg="/sys/fs/cgroup$rel"
  printf 'HOST_MEM_AVAILABLE_KIB=%s\nCGROUP=%s\n' "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" "$cg"
  for name in memory.current memory.peak memory.swap.current memory.swap.peak; do
    test -r "$cg/$name" && printf '%s=%s\n' "$name" "$(cat "$cg/$name")"
  done
  printf '%s\n' 'MEMORY_EVENTS_BEGIN'
  cat "$cg/memory.events"
  printf '%s\n' 'MEMORY_EVENTS_END'
fi

printf '%s\n' 'RESOURCE_TAIL_BEGIN'
tail -n 8 "$RUN/resource.csv" 2>/dev/null || true
printf '%s\n' 'RESOURCE_TAIL_END'
printf 'CHECKPOINT_DIRS=%s\n' "$(find "$RUN" -type d -name 'global_step_*' | wc -l)"
printf 'EVAL_MP4=%s\n' "$(find "$RUN/video/eval" -type f -name '*.mp4' 2>/dev/null | wc -l)"
printf '%s\n' 'GRPO_V2_HIGH_INFO_REFRESH_OK'
