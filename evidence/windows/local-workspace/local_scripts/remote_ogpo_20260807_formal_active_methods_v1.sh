#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

runtime_root=/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260807_v1/runtime
driver_pid=$(cat "$runtime_root/driver_pid.txt")
monitor_pid=$(cat "$runtime_root/monitor_pid.txt")

printf 'STATUS_AT\t%s\n' "$(date --iso-8601=seconds)"
kill -0 "$driver_pid"
kill -0 "$monitor_pid"
printf '%s\n' '=== active OGPO workers ==='
ps -eo pid=,ppid=,stat=,etimes=,rss=,pcpu=,args= --sort=pid \
  | awk '$0 ~ /ray::(EmbodiedOGPOFSDPPolicy|MultiStepRolloutWorker|EnvWorker)/ {print}'
printf '%s\n' '=== gpu compute apps ==='
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits
printf '%s\n' '=== monitor tail ==='
tail -n 3 "$runtime_root/resources_1s.csv"
printf '%s\n' '=== cgroup memory events ==='
cat /sys/fs/cgroup/memory.events
