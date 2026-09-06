#!/usr/bin/env bash
set -u

run_dir=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822
runtime_dir=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822

hostname
date --iso-8601=seconds
for name in wrapper driver observer; do
  pid=$(cat "$runtime_dir/${name}.pid" 2>/dev/null || true)
  if [[ -n "$pid" && -d "/proc/$pid" ]]; then printf '%s=%s:alive\n' "$name" "$pid"; else printf '%s=%s:dead_or_missing\n' "$name" "$pid"; fi
done

printf 'LATEST_GLOBAL_STEPS\n'
if [[ -f "$run_dir/metrics.log" ]]; then
  grep -a 'Global Step' "$run_dir/metrics.log" | tail -n 8 || true
  printf 'METRICS_TAIL\n'
  tail -n 160 "$run_dir/metrics.log"
else
  printf 'metrics.log missing\n'
fi

printf 'DVAC_CSV_TAILS\n'
for csv in "$run_dir"/dvac_train/actor_rank*/runner_step_metrics.csv; do
  printf 'FILE=%s\n' "$csv"
  tail -n 5 "$csv"
done

printf 'ROLLING_STATES\n'
for state in "$run_dir"/dvac_train/actor_rank*/rolling_stats_state.json; do
  printf 'FILE=%s\n' "$state"
  cat "$state"
done

printf 'CHECKPOINTS\n'
find "$run_dir" -type d -name 'global_step_*' -prune -printf '%p\n' | sort -V
printf 'CONTROL_TRACE\n'
find "$run_dir/control_trace" -maxdepth 8 -type f -printf '%p %s\n' 2>/dev/null | sort || true
printf 'INVENTORY\n'
find "$run_dir" -maxdepth 4 -type f -printf '%p %s\n' | sort | tail -n 180
printf 'SIZES\n'
du -sh "$run_dir" "$runtime_dir"
du -sh "$run_dir"/dvac_train "$run_dir"/control_trace "$run_dir"/*/checkpoints 2>/dev/null || true

printf 'GPU\n'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
printf 'CGROUP_CURRENT\n'
cat /sys/fs/cgroup/memory.current
printf 'MEMORY_EVENTS\n'
cat /sys/fs/cgroup/memory.events
printf 'RESOURCE_TAIL\n'
tail -n 3 "$runtime_dir/resource_monitor/resources.csv"
printf 'ERROR_SCAN\n'
grep -aE 'CUDA out of memory|NCCL|WorkerCrashed|RayTaskError|No space left|SIGSEGV|SIGBUS|Killed|all_gather.*(error|Error)|Traceback' "$runtime_dir/driver.log" | tail -n 30 || true
