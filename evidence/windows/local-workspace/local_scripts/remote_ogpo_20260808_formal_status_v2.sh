#!/usr/bin/env bash
set -u
export LC_ALL=C

runtime_root=/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260808_v2/runtime
run_root=/root/autodl-tmp/experiments/ogpo_robotwin_formal_20260808_v2
driver_pid=$(cat "$runtime_root/driver_pid.txt" 2>/dev/null || printf '0')
monitor_pid=$(cat "$runtime_root/monitor_pid.txt" 2>/dev/null || printf '0')

printf 'STATUS_AT\t%s\n' "$(date --iso-8601=seconds)"
kill -0 "$driver_pid" 2>/dev/null && printf 'DRIVER_ALIVE\t1\n' || printf 'DRIVER_ALIVE\t0\n'
kill -0 "$monitor_pid" 2>/dev/null && printf 'MONITOR_ALIVE\t1\n' || printf 'MONITOR_ALIVE\t0\n'
printf 'EXIT_CODE\t%s\n' "$(cat "$runtime_root/exit_code.txt" 2>/dev/null || printf PENDING)"
printf 'RESOURCE_ROWS\t%s\n' "$(($(wc -l <"$runtime_root/resources_1s.csv" 2>/dev/null || printf '1') - 1))"
printf '%s\n' '=== progress ==='
grep -aE 'Evaluating Rollout Epochs|Rollout Epochs|global_step|online_rows|actor_updates|critic_updates' \
  "$runtime_root/driver.log" 2>/dev/null | tail -n 15 || true
printf '%s\n' '=== metrics ==='
find "$run_root" -name metrics.log -type f -exec tail -n 6 {} \; 2>/dev/null || true
printf '%s\n' '=== latest resource ==='
tail -n 1 "$runtime_root/resources_1s.csv" 2>/dev/null || true
printf '%s\n' '=== oom ==='
awk '$1 == "oom" || $1 == "oom_kill" {print}' /sys/fs/cgroup/memory.events
printf '%s\n' '=== gpu ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
