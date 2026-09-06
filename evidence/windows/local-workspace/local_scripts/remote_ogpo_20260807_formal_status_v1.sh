#!/usr/bin/env bash
set -u
export LC_ALL=C

runtime_root=/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260807_v1/runtime
run_root=/root/autodl-tmp/experiments/ogpo_robotwin_formal_20260807_v1
driver_pid=$(cat "$runtime_root/driver_pid.txt" 2>/dev/null || printf '0')
monitor_pid=$(cat "$runtime_root/monitor_pid.txt" 2>/dev/null || printf '0')

printf 'STATUS_AT\t%s\n' "$(date --iso-8601=seconds)"
printf 'DRIVER_PID\t%s\n' "$driver_pid"
if kill -0 "$driver_pid" 2>/dev/null; then printf 'DRIVER_ALIVE\t1\n'; else printf 'DRIVER_ALIVE\t0\n'; fi
printf 'MONITOR_PID\t%s\n' "$monitor_pid"
if kill -0 "$monitor_pid" 2>/dev/null; then printf 'MONITOR_ALIVE\t1\n'; else printf 'MONITOR_ALIVE\t0\n'; fi
printf 'EXIT_CODE\t%s\n' "$(cat "$runtime_root/exit_code.txt" 2>/dev/null || printf 'PENDING')"
printf 'MONITOR_EXIT_CODE\t%s\n' "$(cat "$runtime_root/monitor_exit_code.txt" 2>/dev/null || printf 'PENDING')"
printf 'STARTED_AT\t%s\n' "$(cat "$runtime_root/started_at.txt" 2>/dev/null || printf 'PENDING')"
printf 'FINISHED_AT\t%s\n' "$(cat "$runtime_root/finished_at.txt" 2>/dev/null || printf 'PENDING')"
printf 'RESOURCE_ROWS\t%s\n' "$(($(wc -l <"$runtime_root/resources_1s.csv" 2>/dev/null || printf '1') - 1))"
printf '%s\n' '=== latest resources ==='
tail -n 5 "$runtime_root/resources_1s.csv" 2>/dev/null || true
printf '%s\n' '=== scalar log hints ==='
find "$run_root" -type f \( -name '*.log' -o -name '*.json' -o -name 'events.out.tfevents.*' \) \
  -printf '%s\t%T@\t%p\n' 2>/dev/null | sort -k2,2n | tail -n 30 || true
printf '%s\n' '=== driver tail ==='
tail -n 120 "$runtime_root/driver.log" 2>/dev/null || true
printf '%s\n' '=== output tree ==='
find "$run_root" -maxdepth 5 -type f -printf '%s\t%T@\t%p\n' 2>/dev/null \
  | sort -k2,2n | tail -n 100 || true
printf '%s\n' '=== processes ==='
ps -eo pid,ppid,stat,etimes,rss,pcpu,cmd --sort=pid \
  | grep -E 'train_embodied_agent|raylet|gcs_server|EnvWorker|ActorGroup|RolloutGroup' \
  | grep -v -E 'grep -E|formal_status_v1' || true
printf '%s\n' '=== gpu ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits
printf '%s\n' '=== compute apps ==='
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
printf '%s\n' '=== cgroup memory ==='
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
