#!/usr/bin/env bash
set -u

run=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_global_z_w0to2_formal_100step_2gpu16env_20260823
runtime=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_global_z_w0to2_formal_100step_2gpu16env_20260823
source=/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight

printf 'NOW=%s\n' "$(date --iso-8601=seconds)"
printf 'IDENTITY\n'
hostname
pwd
id -u

printf 'CONTROL_PROCESSES\n'
for name in wrapper driver observer; do
  pid=$(cat "$runtime/$name.pid" 2>/dev/null || true)
  alive=0
  if [ -n "$pid" ] && [ -d "/proc/$pid" ]; then alive=1; fi
  printf '%s_PID=%s ALIVE=%s\n' "${name^^}" "$pid" "$alive"
done
printf 'PGID_PROCESSES\n'
pgid=$(ps -o pgid= -p "$(cat "$runtime/driver.pid" 2>/dev/null || echo 0)" 2>/dev/null | tr -d ' ' || true)
printf 'PGID=%s\n' "$pgid"
if [ -n "$pgid" ]; then
  ps -eo pid,ppid,pgid,etimes,%cpu,%mem,rss,stat,cmd --sort=-rss | awk -v p="$pgid" '$3==p {print}' | head -n 40
fi

printf 'COMPLETE_STEPS\n'
grep -a 'Global Step:' "$run/metrics.log" 2>/dev/null | tail -n 6 || true
printf 'ACTIVE_ROLLOUT\n'
grep -a 'Generating Rollout Epochs:' "$runtime/driver.log" 2>/dev/null | tail -n 5 || true
printf 'EXIT_MARKERS\n'
grep -aE 'FORMAL_DRIVER_RC=|DRIVER_RC=|TRAINING.*(COMPLETE|FINISHED)|100/100' "$runtime"/*.log 2>/dev/null | tail -n 20 || true
printf 'FATAL_MATCHES=%s\n' "$(grep -aEc 'ActorDiedError|RayActorError|CUDA out of memory|OutOfMemory|NCCL.*(error|Error)|fatal|FATAL' "$runtime/driver.log" 2>/dev/null || true)"
printf 'TRACEBACK_COUNT=%s\n' "$(grep -ac 'Traceback' "$runtime/driver.log" 2>/dev/null || true)"
printf 'KNOWN_OPTIONAL_CUROBO_MATCHES=%s\n' "$(grep -ac "No module named 'curobo.types.math'" "$runtime/driver.log" 2>/dev/null || true)"

printf 'CORE_ACTORS\n'
/root/autodl-tmp/RLinf/.venv/bin/ray list actors --detail 2>/dev/null | awk '/class_name: (EmbodiedFSDPActor|MultiStepRolloutWorker|EnvWorker)/ {c=$2} /state:/ && c {print c,$2; c=""}' || true

printf 'ARTIFACT_INVENTORY\n'
du -sh "$run" "$runtime" 2>/dev/null || true
printf 'NPZ_COUNT=%s CSV_COUNT=%s MP4_COUNT=%s PNG_COUNT=%s\n' \
  "$(find "$run" -type f -name '*.npz' 2>/dev/null | wc -l)" \
  "$(find "$run" -type f -name '*.csv' 2>/dev/null | wc -l)" \
  "$(find "$run" -type f -name '*.mp4' 2>/dev/null | wc -l)" \
  "$(find "$run" -type f -name '*.png' 2>/dev/null | wc -l)"
printf 'CHECKPOINTS\n'
find "$run" -maxdepth 5 -type d -name 'global_step_*' -print 2>/dev/null | sort -V
while IFS= read -r d; do
  [ -n "$d" ] || continue
  du -sh "$d"
  find "$d" -maxdepth 4 -type f -name 'complete.json' -print -exec cat {} \;
done < <(find "$run" -maxdepth 5 -type d -name 'global_step_*' -print 2>/dev/null | sort -V)
printf 'LATEST_NPZ\n'
for rank in 00 01; do
  find "$run/dvac_train/actor_rank${rank}" -maxdepth 1 -type f -name 'rollout_step*.npz' -printf '%f %s\n' 2>/dev/null | sort -V | tail -n 4
done
printf 'LATEST_FILE_TIMES\n'
stat -c '%y %s %n' \
  "$run/metrics.log" \
  "$run/dvac_train/actor_rank00/runner_step_metrics.csv" \
  "$run/dvac_train/actor_rank01/runner_step_metrics.csv" \
  "$runtime/driver.log" \
  "$runtime/resource_monitor/resources.csv" 2>/dev/null || true

printf 'GPU\n'
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits
printf 'GPU_COMPUTE\n'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>&1 || true
printf 'CGROUP\n'
printf 'memory.current='; cat /sys/fs/cgroup/memory.current
printf 'memory.max='; cat /sys/fs/cgroup/memory.max
printf 'memory.stat.anon_file='; grep -E '^(anon|file) ' /sys/fs/cgroup/memory.stat | tr '\n' ' '; printf '\n'
grep -E '^max |^oom |^oom_kill ' /sys/fs/cgroup/memory.events
printf 'HOST_MEMORY\n'
free -h
printf 'RESOURCE_FIRST_LAST\n'
{ head -n 2 "$runtime/resource_monitor/resources.csv"; tail -n 2 "$runtime/resource_monitor/resources.csv"; } 2>/dev/null || true
printf 'TOP_RSS_KIB\n'
ps -eo pid,rss,comm,args --sort=-rss | awk 'NR<=12 {print}'
printf 'DISK\n'
df -h /root/autodl-tmp /dev/shm | tail -n +2
printf 'SOURCE\n'
git -C "$source" rev-parse HEAD
git -C "$source" status --short --branch
printf 'RUN_TAIL\n'
tail -n 60 "$runtime/driver.log" 2>/dev/null || true
