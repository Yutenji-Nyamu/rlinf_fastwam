#!/usr/bin/env bash
set -u

runtime_dir=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822
run_dir=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822

printf 'NOW=%s\n' "$(date --iso-8601=seconds)"
for name in wrapper driver observer; do
  pid_file="$runtime_dir/$name.pid"
  if test -f "$pid_file"; then
    pid=$(cat "$pid_file")
    printf '%s_PID=%s ALIVE=%s\n' "${name^^}" "$pid" "$(test -d "/proc/$pid" && echo 1 || echo 0)"
    ps -p "$pid" -o pid=,ppid=,stat=,etime=,rss=,pcpu=,comm=,args= 2>/dev/null || true
  else
    printf '%s_PID=MISSING\n' "${name^^}"
  fi
done
printf 'TARGET_PROCESSES\n'
ps -eo pid=,ppid=,stat=,etime=,rss=,pcpu=,comm=,args= | awk '$5 != "awk" && (index($0,"RLinf_idea2_dvac_residual_downweight") || index($0,"idea2_dvac_r_only_v3_w0to2")) {print}' | tail -n 40
printf 'RUN_DIR=%s\n' "$(test -d "$run_dir" && echo PRESENT || echo ABSENT)"
find "$run_dir" -maxdepth 2 -type f -printf '%p %s\n' 2>/dev/null | sort | tail -n 30
printf 'WRAPPER_TAIL\n'; tail -n 30 "$runtime_dir/wrapper.log" 2>/dev/null || true
printf 'DRIVER_PROGRESS\n'; grep -aE 'Generating Rollout Epochs|Global Step|global_step|Traceback|CUDA out of memory|OutOfMemory|fatal|ERROR' "$runtime_dir/driver.log" 2>/dev/null | tail -n 40 || true
printf 'DRIVER_TAIL\n'; tail -n 50 "$runtime_dir/driver.log" 2>/dev/null || true
printf 'OBSERVER_TAIL\n'; tail -n 20 "$runtime_dir/observer.log" 2>/dev/null || true
printf 'RESOURCE_LAST\n'; tail -n 4 "$runtime_dir/resource_monitor/resources.csv" 2>/dev/null || true
printf 'RESOURCE_LINES=%s\n' "$(wc -l < "$runtime_dir/resource_monitor/resources.csv" 2>/dev/null || echo 0)"
printf 'GPU_STATE\n'; nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits
printf 'GPU_PROCESSES\n'; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader,nounits 2>/dev/null || true
printf 'MEMORY_CURRENT_BYTES=%s\n' "$(cat /sys/fs/cgroup/memory.current)"
printf 'MEMORY_EVENTS\n'; cat /sys/fs/cgroup/memory.events
