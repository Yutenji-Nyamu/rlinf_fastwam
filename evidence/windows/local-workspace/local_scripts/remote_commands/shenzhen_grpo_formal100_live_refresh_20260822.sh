#!/usr/bin/env bash
set -u

RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-formal100-current-4gpu128train64eval-ppo-matched-v1
DRIVER_PID="$(cat "$RUN/driver.pid" 2>/dev/null || true)"
OBSERVER_PID="$(cat "$RUN/resource_observer.pid" 2>/dev/null || true)"

echo '=== TIME_IDENTITY ==='
date --iso-8601=seconds
hostname
id
printf 'run=%s\ndriver_pid=%s\nobserver_pid=%s\n' "$RUN" "$DRIVER_PID" "$OBSERVER_PID"

echo '=== OWNED_TOP_LEVEL_PROCESSES ==='
for pid in "$DRIVER_PID" "$OBSERVER_PID"; do
  [ -n "$pid" ] || continue
  if [ -d "/proc/$pid" ]; then
    ps -o user=,pid=,ppid=,pgid=,etimes=,rss=,stat=,comm=,args= -p "$pid"
  else
    printf 'pid=%s alive=false\n' "$pid"
  fi
done

echo '=== RAY_PROCESS_COUNTS_AND_RSS ==='
ps -u chenyiteng -o pid=,ppid=,etimes=,rss=,stat=,comm=,args= | awk '
  /ray::EnvWorker/ {c_env++; rss_env+=$4}
  /ray::ActorRolloutRefWorker/ {c_ar++; rss_ar+=$4}
  /ray::RolloutWorker/ {c_roll++; rss_roll+=$4}
  /gcs_server/ {c_gcs++; rss_gcs+=$4}
  /raylet/ {c_raylet++; rss_raylet+=$4}
  END {
    printf "EnvWorker count=%d rss_kib=%d\n", c_env, rss_env;
    printf "ActorRolloutRefWorker count=%d rss_kib=%d\n", c_ar, rss_ar;
    printf "RolloutWorker count=%d rss_kib=%d\n", c_roll, rss_roll;
    printf "gcs_server count=%d rss_kib=%d\n", c_gcs, rss_gcs;
    printf "raylet count=%d rss_kib=%d\n", c_raylet, rss_raylet;
  }'

echo '=== PROGRESS_LINES ==='
grep -aE 'Global Step:|Generating Rollout Epochs:|Evaluating|Saving|checkpoint|success rate|Success Rate' "$RUN/driver.log" 2>/dev/null | tail -n 100 || true

echo '=== LATEST_COMPLETE_METRIC_TABLE ==='
grep -a 'Global Step:' "$RUN/driver.log" 2>/dev/null | tail -n 3 || true

echo '=== GRPO_SIGNAL_LINES ==='
grep -aEi 'grpo|group|filter|advantage|loss_mask|homogeneous|policy_loss|actor_loss|approx_kl|clip_fraction|grad_norm|entropy|return|success' "$RUN/driver.log" 2>/dev/null | tail -n 220 || true

echo '=== DRIVER_TAIL ==='
tail -n 220 "$RUN/driver.log" 2>/dev/null || true

echo '=== FATAL_MATCHES ==='
grep -aEin 'Traceback|CUDA out of memory|OutOfMemoryError|WorkerCrashedError|SIGKILL|Killed process|NCCL.*(error|timeout)|RayTaskError|illegal instruction|segmentation fault|fatal|exception|error:' "$RUN/driver.log" 2>/dev/null | tail -n 100 || true

echo '=== RUN_FILES ==='
stat -c 'size=%s mtime=%y path=%n' "$RUN/driver.log" "$RUN/metrics.log" "$RUN/resource.csv" "$RUN/resolved.yaml" 2>/dev/null || true
find "$RUN" -maxdepth 5 -type f \( -name 'events.out.tfevents*' -o -name '*.tfevents' -o -name '*.mp4' -o -name 'full_weights.pt' \) -printf '%s %T@ %p\n' 2>/dev/null | sort -n

echo '=== CHECKPOINTS_AND_EVAL ==='
for d in "$RUN"/robotwin_grpo_openpi/checkpoints/global_step_* "$RUN"/checkpoints/global_step_*; do
  [ -d "$d" ] || continue
  du -sb "$d"
  find "$d" -maxdepth 4 -type f -printf '%s %p\n' 2>/dev/null | sort -n | tail -n 30
done
printf 'train_mp4_count='; find "$RUN/video/train" -type f -name '*.mp4' 2>/dev/null | wc -l
printf 'eval_mp4_count='; find "$RUN/video/eval" -type f -name '*.mp4' 2>/dev/null | wc -l
find "$RUN/video/eval" -mindepth 1 -maxdepth 3 -type d -printf '%p\n' 2>/dev/null | sort

echo '=== RESOURCE_HEAD_TAIL ==='
head -n 3 "$RUN/resource.csv" 2>/dev/null || true
tail -n 12 "$RUN/resource.csv" 2>/dev/null || true

echo '=== GPU_4_7 ==='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu --format=csv,noheader,nounits | awk -F, '$1+0 >= 4 && $1+0 <= 7 {print}'
echo '=== GPU_COMPUTE_APPS ==='
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader || true

echo '=== HOST_AND_CGROUP_MEMORY ==='
free -h
awk '/^(MemTotal|MemAvailable|AnonPages|Cached|Shmem|SwapTotal|SwapFree):/ {print $1,$2}' /proc/meminfo
if [ -n "$DRIVER_PID" ] && [ -r "/proc/$DRIVER_PID/cgroup" ]; then
  REL="$(awk -F: '$1=="0" {print $3}' "/proc/$DRIVER_PID/cgroup")"
  CG="/sys/fs/cgroup$REL"
  printf 'cgroup_path=%s\n' "$CG"
  for f in memory.current memory.peak memory.swap.current memory.swap.peak; do
    [ -r "$CG/$f" ] && { printf '%s=' "$f"; cat "$CG/$f"; }
  done
  [ -r "$CG/memory.events" ] && cat "$CG/memory.events"
fi

echo '=== RESOLVED_HASH ==='
sha256sum "$RUN/resolved.yaml" 2>/dev/null || true
cat "$RUN/resolved.yaml.sha256" 2>/dev/null || true
echo 'SZ_GRPO_FORMAL100_LIVE_REFRESH_DONE'
