#!/usr/bin/env bash
set -u
export LC_ALL=C

runtime_root=/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260807_v1/runtime
run_root=/root/autodl-tmp/experiments/ogpo_robotwin_formal_20260807_v1
driver_pid=$(cat "$runtime_root/driver_pid.txt" 2>/dev/null || printf '0')
monitor_pid=$(cat "$runtime_root/monitor_pid.txt" 2>/dev/null || printf '0')

printf 'STATUS_AT\t%s\n' "$(date --iso-8601=seconds)"
if kill -0 "$driver_pid" 2>/dev/null; then printf 'DRIVER_ALIVE\t1\n'; else printf 'DRIVER_ALIVE\t0\n'; fi
if kill -0 "$monitor_pid" 2>/dev/null; then printf 'MONITOR_ALIVE\t1\n'; else printf 'MONITOR_ALIVE\t0\n'; fi
printf 'EXIT_CODE\t%s\n' "$(cat "$runtime_root/exit_code.txt" 2>/dev/null || printf 'PENDING')"
printf 'RESOURCE_ROWS\t%s\n' "$(($(wc -l <"$runtime_root/resources_1s.csv" 2>/dev/null || printf '1') - 1))"
printf '%s\n' '=== progress markers ==='
grep -aE 'Evaluating Rollout Epochs|Rollout Epochs|global_step|online_rows|total_online_rows|actor_updates|critic_updates' \
  "$runtime_root/driver.log" 2>/dev/null | tail -n 30 || true
printf '%s\n' '=== fatal markers ==='
grep -aE 'Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|ActorDiedError|NaN|Inf' \
  "$runtime_root/driver.log" 2>/dev/null | tail -n 30 || true
printf '%s\n' '=== latest resources ==='
tail -n 5 "$runtime_root/resources_1s.csv" 2>/dev/null || true
printf '%s\n' '=== gpu ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw \
  --format=csv,noheader,nounits
printf '%s\n' '=== cgroup memory events ==='
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
printf '%s\n' '=== output files ==='
find "$run_root" -maxdepth 5 -type f -printf '%s\t%T@\t%p\n' 2>/dev/null \
  | sort -k2,2n | tail -n 30 || true
