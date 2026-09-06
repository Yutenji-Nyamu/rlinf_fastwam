#!/usr/bin/env bash
set -u

RUN=/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-formal100-4gpu128train64eval-official-v1
DRIVER_PID=1375834
LOG="$RUN/driver.log"

echo '=== AUDIT_TIME ==='
date --iso-8601=seconds
hostname
id

echo '=== DRIVER_PID ==='
if [ -r "/proc/$DRIVER_PID/cmdline" ]; then
  ps -p "$DRIVER_PID" -o pid=,ppid=,stat=,lstart=,etime=,%cpu=,%mem=,rss=,vsz=,cmd=
  tr '\0' ' ' < "/proc/$DRIVER_PID/cmdline"
  echo
else
  echo "driver_pid_not_alive=$DRIVER_PID"
fi

echo '=== OWNED_PROJECT_PROCESSES ==='
ps -u "$(id -u)" -o pid=,ppid=,stat=,etime=,%cpu=,%mem=,rss=,cmd= --sort=pid \
  | grep -E 'ppo-formal100-4gpu128train64eval-official-v1|train_embodied_agent.py|raylet|gcs_server|dashboard.py|default_worker.py|ray::|RoboTwin' \
  | grep -v grep || true

echo '=== GPU_SNAPSHOT ==='
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit \
  --format=csv,noheader,nounits

echo '=== GPU_COMPUTE_APPS ==='
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true

echo '=== HOST_MEMORY_AND_LOAD ==='
free -b
cat /proc/loadavg
df -B1 /home /data

echo '=== RUN_ROOT ==='
if [ -d "$RUN" ]; then
  du -sh "$RUN"
  find "$RUN" -maxdepth 3 -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' | sort
else
  echo "run_missing=$RUN"
fi

echo '=== DRIVER_LOG_STAT ==='
if [ -f "$LOG" ]; then
  stat -c 'size=%s mtime=%y path=%n' "$LOG"
  echo '=== DRIVER_LOG_PROGRESS_AND_METRICS ==='
  grep -aE 'Global Step|Generating Rollout|success_once|num_trajectories|actor/|critic/|eval/|policy_loss|value_loss|total_loss|grad_norm|approx_kl|clip_fraction|advantages|returns|reward|Saving|checkpoint|Finished|Traceback|CUDA out of memory|NCCL|SIGKILL|Killed|ERROR' "$LOG" | tail -n 500 || true
  echo '=== DRIVER_LOG_TAIL ==='
  tail -n 180 "$LOG"
else
  echo "driver_log_missing=$LOG"
fi

echo '=== CHECKPOINTS ==='
CKPT="$RUN/robotwin_ppo_openpi/checkpoints"
if [ -d "$CKPT" ]; then
  find "$CKPT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -V
  du -sh "$CKPT"/* 2>/dev/null || true
  find "$CKPT" -maxdepth 5 -type f -printf '%s %p\n' | sort -k2
else
  echo 'no_checkpoint_directory'
fi

echo '=== VIDEOS ==='
if [ -d "$RUN/video" ]; then
  find "$RUN/video" -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' | sort
  du -sh "$RUN/video"
else
  echo 'no_video_directory'
fi

echo '=== EVENT_AND_METRIC_FILES ==='
find "$RUN" -type f \( -name 'events.out.tfevents*' -o -name '*.jsonl' -o -name '*.csv' -o -name '*metric*' -o -name '*resource*' \) \
  -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' | sort || true

echo '=== FATAL_COUNTS ==='
if [ -f "$LOG" ]; then
  printf 'traceback='; grep -ac 'Traceback' "$LOG" || true
  printf 'cuda_oom='; grep -aci 'CUDA out of memory' "$LOG" || true
  printf 'nccl_error='; grep -aciE 'NCCL.*(error|fail|timeout)' "$LOG" || true
  printf 'worker_died='; grep -aciE 'worker.*(died|crash)|RayActorError|WorkerCrashedError' "$LOG" || true
  printf 'nan_inf='; grep -aciE '(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)' "$LOG" || true
fi

echo 'FORMAL100_LIVE_AUDIT_DONE'
