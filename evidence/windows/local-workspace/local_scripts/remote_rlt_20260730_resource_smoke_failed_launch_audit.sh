#!/usr/bin/env bash
set -u

runtime=/root/autodl-tmp/experiment_exports/rlt_stage2_resource_smoke_8env3c_20260730_v1/runtime
run_root=/root/autodl-tmp/experiments/rlt_stage2_resource_smoke_8env3c_20260730_v1
printf 'time\t%s\n' "$(date --iso-8601=seconds)"
printf 'run_exists\t%s\n' "$(test -e "$run_root" && echo yes || echo no)"
printf 'files_begin\n'
find "$runtime" -maxdepth 1 -type f -printf '%f\t%s\n' | sort
printf 'files_end\n'
printf 'driver_log_begin\n'
tail -n 160 "$runtime/driver.log" 2>/dev/null || true
printf 'driver_log_end\n'
printf 'exit_code\t%s\n' "$(cat "$runtime/exit_code.txt" 2>/dev/null || echo absent)"
printf 'started_at\t%s\n' "$(cat "$runtime/started_at.txt" 2>/dev/null || echo absent)"
printf 'finished_at\t%s\n' "$(cat "$runtime/finished_at.txt" 2>/dev/null || echo absent)"
printf 'run_foreground_begin\n'
sed -n '1,120p' "$runtime/run_foreground.sh"
printf 'run_foreground_end\n'
printf 'process_begin\n'
ps -eo pid=,comm=,args= \
  | awk '$2 ~ /^python/ && $0 ~ /train_embodied_agent[.]py/ {print}'
pgrep -ax raylet || true
pgrep -ax gcs_server || true
printf 'process_end\n'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits
