#!/usr/bin/env bash
set -u

runtime_dir=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822
run_dir=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822

date --iso-8601=seconds
for name in wrapper driver observer; do
  pid_file="$runtime_dir/${name}.pid"
  if [[ -f "$pid_file" ]]; then
    pid=$(cat "$pid_file")
    if [[ -d "/proc/$pid" ]]; then
      printf '%s_PID=%s ALIVE=1 CMD=' "$name" "$pid"
      tr '\0' ' ' < "/proc/$pid/cmdline"
      printf '\n'
    else
      printf '%s_PID=%s ALIVE=0\n' "$name" "$pid"
    fi
  else
    printf '%s_PID_FILE_MISSING\n' "$name"
  fi
done

printf 'RAY_WORKERS\n'
ps -eo pid,ppid,pgid,stat,cmd --sort=pid | grep -E 'ray::|raylet|train_embodied_agent' | grep -v grep | tail -n 30 || true
printf 'GPU\n'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
printf 'CGROUP_CURRENT\n'
cat /sys/fs/cgroup/memory.current
printf 'MEMORY_EVENTS\n'
cat /sys/fs/cgroup/memory.events
printf 'RUNTIME_FILES\n'
find "$runtime_dir" -maxdepth 2 -type f -printf '%P %s\n' | sort
printf 'RUN_FILES\n'
if [[ -d "$run_dir" ]]; then find "$run_dir" -maxdepth 3 -type f -printf '%P %s\n' | sort | head -n 80; else printf 'RUN_DIR_MISSING\n'; fi
printf 'DRIVER_MILESTONES\n'
if [[ -f "$runtime_dir/driver.log" ]]; then
  grep -E 'Global Step|Generating Rollout Epochs|Rollout|Actor|EnvGroup|checkpoint|Loaded|Loading|Traceback|CUDA out of memory|NCCL|WorkerCrashed|RayTaskError|No space left|SIGSEGV|SIGBUS|Killed|all_gather' "$runtime_dir/driver.log" | tail -n 120 || true
  printf 'DRIVER_TAIL\n'
  tail -n 80 "$runtime_dir/driver.log"
fi
