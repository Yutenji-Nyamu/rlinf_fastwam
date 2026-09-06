set -u

RUN=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822
RUNTIME=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822

echo '=== TIME_IDENTITY ==='
date '+%Y-%m-%d %H:%M:%S %Z'
hostname
id -u

echo '=== CONTROL_PIDS ==='
for role in wrapper driver observer; do
  pid_file="$RUNTIME/$role.pid"
  if [ -f "$pid_file" ]; then
    pid=$(cat "$pid_file")
    echo "$role pid=$pid"
    ps -p "$pid" -o pid=,ppid=,pgid=,stat=,etime=,%cpu=,%mem=,rss=,cmd= || true
  else
    echo "$role pid_file_missing"
  fi
done

echo '=== RUN_MATCHING_PROCESSES ==='
pgrep -af 'idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822|ray::ActorWorker|ray::RolloutWorker|ray::EnvWorker' | head -n 40 || true

echo '=== GPU ==='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits
echo '--- GPU COMPUTE PROCESSES ---'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true

echo '=== CGROUP_MEMORY ==='
printf 'memory.current='; cat /sys/fs/cgroup/memory.current
printf 'memory.max='; cat /sys/fs/cgroup/memory.max
printf 'memory.events='; tr '\n' ' ' < /sys/fs/cgroup/memory.events; echo
for key in anon file shmem inactive_file active_file; do
  awk -v k="$key" '$1==k {print k"="$2}' /sys/fs/cgroup/memory.stat
done
free -h

echo '=== TRAINING_PROGRESS ==='
if [ -f "$RUN/metrics.log" ]; then
  grep -a 'Global Step:' "$RUN/metrics.log" | tail -n 5
  echo '--- LATEST METRIC TABLE ---'
  tail -n 42 "$RUN/metrics.log"
else
  echo 'metrics.log missing'
fi

echo '=== ROLLOUT_PROGRESS ==='
if [ -f "$RUNTIME/driver.log" ]; then
  grep -aE 'Rollout epoch|Global Step|save|checkpoint' "$RUNTIME/driver.log" | tail -n 30 || true
  echo '--- DRIVER TAIL ---'
  tail -n 45 "$RUNTIME/driver.log"
  echo '--- FATAL COUNTS ---'
  for p in 'CUDA out of memory' 'OutOfMemory' 'oom_kill' 'ActorDiedError' 'RayActorError' 'NCCL' 'Traceback'; do
    c=$(grep -aic "$p" "$RUNTIME/driver.log" || true)
    echo "$p=$c"
  done
else
  echo 'driver.log missing'
fi

echo '=== DVAC_LATEST ==='
for rank in 00 01; do
  csv="$RUN/dvac_train/actor_rank${rank}/runner_step_metrics.csv"
  echo "--- actor_rank${rank} ---"
  if [ -f "$csv" ]; then
    head -n 1 "$csv"
    tail -n 3 "$csv"
  else
    echo 'missing'
  fi
done

echo '=== CHECKPOINTS ==='
find "$RUN/checkpoints" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' 2>/dev/null | sort -V || true
du -sh "$RUN" "$RUNTIME" 2>/dev/null || true
find "$RUN" -maxdepth 3 -type f -printf '%s %p\n' 2>/dev/null | sort -nr | head -n 20 || true

echo '=== ARTIFACT_COUNTS ==='
printf 'dvac_npz='; find "$RUN/dvac_train" -type f -name 'rollout_step*.npz' 2>/dev/null | wc -l
printf 'dvac_csv='; find "$RUN/dvac_train" -type f -name '*.csv' 2>/dev/null | wc -l
printf 'control_mp4='; find "$RUN/control_trace" -type f -name '*.mp4' 2>/dev/null | wc -l
printf 'control_frames_csv='; find "$RUN/control_trace" -type f -name 'frames.csv' 2>/dev/null | wc -l
printf 'tensorboard_events='; find "$RUN/tensorboard" -type f -name 'events.out.tfevents*' 2>/dev/null | wc -l
find "$RUN/control_trace" -type f -printf '%s %p\n' 2>/dev/null | sort -nr | head -n 12 || true

echo '=== RESOURCE_MONITOR_TAIL ==='
RES="$RUNTIME/resource_monitor/resources.csv"
if [ -f "$RES" ]; then
  head -n 1 "$RES"
  tail -n 6 "$RES"
  wc -l "$RES"
else
  echo 'resources.csv missing'
fi

echo '=== SPACE ==='
df -h /root/autodl-tmp
df -i /root/autodl-tmp
