#!/usr/bin/env bash
set -u

run=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_apply_formal_100step_2gpu16env_20260821
runtime=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_apply_formal_100step_2gpu16env_20260821

printf 'identity\n'
hostname
pwd
id -u
date --iso-8601=seconds
printf 'processes\n'
ps -p 114146,114149,114150 -o pid=,stat=,etimes=,cmd=
printf 'latest_steps\n'
grep -o 'Global Step:[[:space:]]*[0-9]\+/100' "$run/metrics.log" | tail -n 3
printf 'latest_metric_tail\n'
tail -n 65 "$run/metrics.log"
printf 'checkpoints\n'
find "$run/checkpoints" -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' 2>/dev/null | sort -V | tail -n 6
printf 'gpu\n'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
printf 'cgroup\n'
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
printf 'completion\n'
for f in driver.exitcode launch_finished_at.txt; do
  if [[ -f "$runtime/$f" ]]; then
    printf '%s=' "$f"
    cat "$runtime/$f"
  else
    printf '%s=missing\n' "$f"
  fi
done
