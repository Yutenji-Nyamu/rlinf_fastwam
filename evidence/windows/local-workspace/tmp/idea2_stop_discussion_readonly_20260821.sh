#!/usr/bin/env bash
set -u

run=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_apply_formal_100step_2gpu16env_20260821
runtime=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_apply_formal_100step_2gpu16env_20260821

printf 'identity\n'
hostname
pwd
id -u
date --iso-8601=seconds

printf 'owned_processes\n'
ps -p 114146,114149,114150 -o pid=,ppid=,pgid=,sid=,stat=,etimes=,cmd= || true
pgrep -af 'train_embodied_agent.py.*robotwin_adjust_bottle_grpo_openpi_dvac_train_100step_formal' || true

printf 'latest_step\n'
grep -o 'Global Step:[[:space:]]*[0-9]\+/100' "$run/metrics.log" | tail -n 3
printf 'latest_metrics\n'
tail -n 75 "$run/metrics.log"

printf 'completion_markers\n'
for f in driver.exitcode launch_finished_at.txt wrapper.pid driver.pid observer.pid; do
  if [[ -f "$runtime/$f" ]]; then
    printf '%s=' "$f"
    cat "$runtime/$f"
  else
    printf '%s=missing\n' "$f"
  fi
done

printf 'checkpoint_inventory\n'
find "$run/checkpoints" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort -V
du -sh "$run/checkpoints" "$run/dvac_train" "$run/control_trace" "$run/video" "$run" 2>/dev/null || true

printf 'small_artifact_inventory\n'
find "$run" -maxdepth 2 -type f \( -name '*.log' -o -name '*.yaml' -o -name '*.json' -o -name '*.csv' -o -name '*.png' -o -name '*.mp4' \) -printf '%s %p\n' 2>/dev/null | sort -n | tail -n 80

printf 'gpu_and_memory\n'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
printf 'memory_current='; cat /sys/fs/cgroup/memory.current
printf 'memory_events='; tr '\n' ' ' </sys/fs/cgroup/memory.events; printf '\n'
df -h /root/autodl-tmp

printf 'resource_tail\n'
tail -n 5 "$runtime/resources.csv" 2>/dev/null || true
