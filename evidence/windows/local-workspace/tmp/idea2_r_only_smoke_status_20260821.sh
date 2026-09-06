#!/usr/bin/env bash
set -u

runtime_dir=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_downweight_smoke_2step_2gpu16env_20260821
run_dir=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_downweight_smoke_2step_2gpu16env_20260821

printf 'PIDS\n'
for name in wrapper driver observer; do
  pid=$(cat "$runtime_dir/$name.pid" 2>/dev/null || true)
  if test -n "$pid" && test -d "/proc/$pid"; then
    printf '%s=%s alive\n' "$name" "$pid"
  else
    printf '%s=%s exited\n' "$name" "$pid"
  fi
done

printf 'EXIT_MARKERS\n'
for item in driver.exitcode observer.exitcode launch_finished_at.txt; do
  if test -e "$runtime_dir/$item"; then
    printf '%s=' "$item"
    cat "$runtime_dir/$item"
  else
    printf '%s=absent\n' "$item"
  fi
done

printf 'RESOURCES\n'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
wc -l "$runtime_dir/resource_monitor/resources.csv" 2>/dev/null || true

printf 'RUN_FILES\n'
find "$run_dir" -maxdepth 3 -type f 2>/dev/null | sort | tail -n 30

printf 'LOG_SIGNALS\n'
grep -E 'Global Step|Model initialized|Actor|Rollout|EnvWorker|Traceback|CUDA out of memory|OutOfMemoryError|worker died|fatal' \
  "$runtime_dir/driver.log" 2>/dev/null | tail -n 40 || true
printf 'LOG_TAIL\n'
tail -n 50 "$runtime_dir/driver.log" 2>/dev/null || true
