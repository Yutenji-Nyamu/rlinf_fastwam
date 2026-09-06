#!/usr/bin/env bash
set -u

RUN=/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-formal100-4gpu128train64eval-official-v1
LOG="$RUN/driver.log"
METRICS="$RUN/metrics.log"
PID=1375834

echo '=== TIME_PID ==='
date --iso-8601=seconds
if [ -r "/proc/$PID/stat" ]; then
  ps -p "$PID" -o pid=,ppid=,stat=,lstart=,etime=,%cpu=,%mem=,rss=,vsz=
else
  echo "not_alive=$PID"
fi

echo '=== PROCESS_COUNTS ==='
printf 'user_processes='; ps -u "$(id -u)" --no-headers | wc -l
printf 'ray_named='; ps -u "$(id -u)" -o comm= | grep -c '^ray::' || true
printf 'raylet='; ps -u "$(id -u)" -o comm= | grep -c '^raylet$' || true
printf 'gcs_server='; ps -u "$(id -u)" -o comm= | grep -c '^gcs_server$' || true

echo '=== GPU ==='
nvidia-smi --query-gpu=index,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit \
  --format=csv,noheader,nounits

echo '=== GPU_APPS_BY_INDEX ==='
nvidia-smi pmon -c 1 -s um | sed -n '1,20p'

echo '=== MEMORY_LOAD_DISK ==='
free -h
cat /proc/loadavg
df -h /home /data

echo '=== RUN_SUMMARY ==='
du -sh "$RUN" 2>/dev/null || true
find "$RUN" -mindepth 1 -maxdepth 1 -printf '%y %f\n' | sort
stat -c 'driver_log_bytes=%s driver_log_mtime=%y' "$LOG" 2>/dev/null || true
stat -c 'metrics_log_bytes=%s metrics_log_mtime=%y' "$METRICS" 2>/dev/null || true

echo '=== METRICS_HEAD ==='
head -n 5 "$METRICS" 2>/dev/null || true
echo '=== METRICS_TAIL ==='
tail -n 30 "$METRICS" 2>/dev/null || true
echo '=== METRICS_LINES ==='
wc -l "$METRICS" 2>/dev/null || true

echo '=== DRIVER_PROGRESS_TAIL ==='
grep -aE 'Global Step|Generating Rollout Epochs|success_once|num_trajectories|policy_loss|value_loss|total_loss|grad_norm|approx_kl|clip_fraction|explained_variance|Saving|checkpoint|Eval|eval/' "$LOG" \
  | tail -n 240 || true

echo '=== DRIVER_RAW_TAIL ==='
tail -n 100 "$LOG" 2>/dev/null || true

echo '=== CHECKPOINT_SUMMARY ==='
CKPT="$RUN/robotwin_ppo_openpi/checkpoints"
if [ -d "$CKPT" ]; then
  find "$CKPT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -V
  du -sh "$CKPT"/* 2>/dev/null || true
else
  echo none
fi

echo '=== VIDEO_SUMMARY ==='
for kind in train eval; do
  root="$RUN/video/$kind"
  if [ -d "$root" ]; then
    printf '%s_count=' "$kind"; find "$root" -type f -name '*.mp4' | wc -l
    printf '%s_bytes=' "$kind"; find "$root" -type f -name '*.mp4' -printf '%s\n' | awk '{s+=$1} END{print s+0}'
    for seed in 0 1 2 3; do
      printf '%s_seed_%s_latest=' "$kind" "$seed"
      find "$root/seed_$seed" -maxdepth 1 -type f -name '*.mp4' -printf '%f\n' 2>/dev/null | sort -V | tail -n 1
    done
  else
    echo "${kind}_none"
  fi
done

echo '=== TENSORBOARD_SUMMARY ==='
find "$RUN/tensorboard" -maxdepth 1 -type f -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %f\n' 2>/dev/null | sort || true

echo '=== FATAL_COUNTS ==='
printf 'traceback='; grep -ac 'Traceback' "$LOG" 2>/dev/null || true
printf 'cuda_oom='; grep -aci 'CUDA out of memory' "$LOG" 2>/dev/null || true
printf 'nccl_error='; grep -aciE 'NCCL.*(error|fail|timeout)' "$LOG" 2>/dev/null || true
printf 'worker_died='; grep -aciE 'worker.*(died|crash)|RayActorError|WorkerCrashedError' "$LOG" 2>/dev/null || true
printf 'nan_inf='; grep -aciE '(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)' "$LOG" 2>/dev/null || true

echo 'FORMAL100_COMPACT_AUDIT_DONE'
