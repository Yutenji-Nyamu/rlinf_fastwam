set -u

run=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_apply_formal_100step_2gpu16env_20260821
runtime=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_apply_formal_100step_2gpu16env_20260821

echo '=== IDENTITY_TIME ==='
date -Is
hostname
pwd
id -u

echo '=== CONTROLLER_STATE ==='
for name in wrapper driver observer; do
  pid_file="$runtime/$name.pid"
  if test -f "$pid_file"; then
    pid=$(cat "$pid_file")
    printf '%s_pid=%s\n' "$name" "$pid"
    ps -o pid,ppid,stat,rss,etimes,args -p "$pid" || true
  else
    printf '%s_pid_file=missing\n' "$name"
  fi
done
for file in driver.exitcode observer.exitcode launch_started_at.txt launch_finished_at.txt; do
  if test -f "$runtime/$file"; then
    printf '%s=' "$file"
    cat "$runtime/$file"
  else
    printf '%s=missing\n' "$file"
  fi
done

echo '=== ACTIVE_WORKERS ==='
ps -eo pid,ppid,stat,rss,etimes,args --sort=-rss | \
  grep -E 'train_embodied_agent|ray::(EnvWorker|EmbodiedF|MultiStepR|ChannelW|Collective)' | \
  grep -v grep | head -40 || true

echo '=== GPU_MEMORY ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw \
  --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,name --format=csv,noheader || true

echo '=== MEMORY_DISK ==='
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
awk '/MemAvailable|MemTotal/ {print}' /proc/meminfo
df -h /root/autodl-tmp /dev/shm
du -sh "$run" 2>/dev/null || true

echo '=== METRICS_FILES ==='
find "$run" -maxdepth 3 -type f \
  \( -name 'metrics.log' -o -name 'runner_step_metrics.csv' -o -name 'rolling_stats_state.json' \
     -o -name 'run_manifest.json' -o -name 'events.out.tfevents.*' \) \
  -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort

echo '=== CHECKPOINTS ==='
find "$run" -maxdepth 5 -type d -name 'global_step_*' -printf '%TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort
find "$run" -maxdepth 5 -type d -name 'global_step_*' -exec du -sh {} \; 2>/dev/null | sort -V || true

echo '=== METRICS_LOG_FULL ==='
if test -f "$run/metrics.log"; then
  wc -l -c "$run/metrics.log"
  cat "$run/metrics.log"
fi

echo '=== ACTOR_STEP_CSV_FULL ==='
for file in "$run"/dvac_train/actor_rank*/runner_step_metrics.csv; do
  test -f "$file" || continue
  echo "--- $file"
  wc -l -c "$file"
  cat "$file"
done

echo '=== DVAC_SHARDS ==='
for dir in "$run"/dvac_train/actor_rank*; do
  test -d "$dir" || continue
  echo "--- $dir"
  find "$dir" -maxdepth 1 -type f -printf '%f %s\n' | sort
done

echo '=== CONTROL_TRACE ==='
find "$run/control_trace" -maxdepth 8 -type f -printf '%p %s\n' 2>/dev/null | sort || true
find "$run/control_trace" -type f -name metadata.json -exec sh -c 'echo --- "$1"; cat "$1"' sh {} \; 2>/dev/null || true

echo '=== RESOURCE_MONITOR ==='
for file in resources.csv process_rss.tsv observer_exit.txt; do
  if test -f "$runtime/resource_monitor/$file"; then
    wc -l -c "$runtime/resource_monitor/$file"
    tail -n 8 "$runtime/resource_monitor/$file"
  fi
done

echo '=== DRIVER_TAIL ==='
wc -l -c "$runtime/driver.log" 2>/dev/null || true
tail -n 180 "$runtime/driver.log" 2>/dev/null || true

echo '=== ERROR_SCAN ==='
grep -E 'CUDA out of memory|OutOfMemory|worker died|WorkerCrashed|NCCL.*(error|Error)|oom_kill|RayTaskError|Traceback|Error' \
  "$runtime/driver.log" 2>/dev/null | tail -80 || true
