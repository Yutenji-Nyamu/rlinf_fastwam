#!/usr/bin/env bash
set -u

driver_pid=$1
output_csv=$2
interval_seconds=${3:-1}

printf '%s\n' \
  'unix_time,host_available_bytes,cgroup_current_bytes,cgroup_anon_bytes,cgroup_file_bytes,cgroup_oom_events,cgroup_oom_kill_events,shm_used_bytes,disk_available_bytes,gpu0_used_mib,gpu0_total_mib,gpu0_util_pct,gpu0_mem_util_pct,gpu0_power_w,gpu1_used_mib,gpu1_total_mib,gpu1_util_pct,gpu1_mem_util_pct,gpu1_power_w,env_rss_kib,env_cpu_pct,actor_rss_kib,actor_cpu_pct,rollout_rss_kib,rollout_cpu_pct,driver_rss_kib,driver_cpu_pct,ray_rss_kib,ray_cpu_pct,matched_total_rss_kib,compute_process_count' \
  >"$output_csv"

memory_event() {
  awk -v key="$1" '$1 == key {print $2}' /sys/fs/cgroup/memory.events 2>/dev/null \
    || printf '%s' -1
}

process_metric() {
  include_pattern=$1
  field=$2
  ps -eo rss=,pcpu=,args= | awk -v include="$include_pattern" -v field="$field" \
    '$0 ~ include && $0 !~ /smoke_monitor_v1|awk -v include=/ {
       if (field == "rss") total += $1; else total += $2
     } END {printf "%.2f", total + 0}'
}

sample_once() {
  now=$(date +%s)
  host_available=$(awk '/MemAvailable:/ {print $2 * 1024}' /proc/meminfo)
  cgroup_current=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || printf '%s' -1)
  cgroup_anon=$(awk '$1 == "anon" {print $2}' /sys/fs/cgroup/memory.stat 2>/dev/null || printf '%s' -1)
  cgroup_file=$(awk '$1 == "file" {print $2}' /sys/fs/cgroup/memory.stat 2>/dev/null || printf '%s' -1)
  cgroup_oom=$(memory_event oom)
  cgroup_oom_kill=$(memory_event oom_kill)
  shm_used=$(df -B1 --output=used /dev/shm 2>/dev/null | tail -n 1 | tr -d ' ')
  disk_available=$(df -B1 --output=avail /root/autodl-tmp 2>/dev/null | tail -n 1 | tr -d ' ')
  gpu_rows=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits 2>/dev/null || true)
  gpu0=$(printf '%s\n' "$gpu_rows" | awk -F, '$1 + 0 == 0 {gsub(/ /, ""); print $2","$3","$4","$5","$6}')
  gpu1=$(printf '%s\n' "$gpu_rows" | awk -F, '$1 + 0 == 1 {gsub(/ /, ""); print $2","$3","$4","$5","$6}')
  env_rss=$(process_metric 'EnvWorker' rss)
  env_cpu=$(process_metric 'EnvWorker' cpu)
  actor_rss=$(process_metric 'EmbodiedOGPOFSDPPolicy|ActorGroup' rss)
  actor_cpu=$(process_metric 'EmbodiedOGPOFSDPPolicy|ActorGroup' cpu)
  rollout_rss=$(process_metric 'MultiStepRolloutWorker|RolloutGroup' rss)
  rollout_cpu=$(process_metric 'MultiStepRolloutWorker|RolloutGroup' cpu)
  driver_rss=$(ps -o rss= -p "$driver_pid" 2>/dev/null | awk 'NF {print $1; found=1} END {if (!found) print 0}')
  driver_cpu=$(ps -o pcpu= -p "$driver_pid" 2>/dev/null | awk 'NF {print $1; found=1} END {if (!found) print 0}')
  ray_rss=$(process_metric 'raylet|gcs_server|dashboard.py|dashboard/agent.py|log_monitor.py' rss)
  ray_cpu=$(process_metric 'raylet|gcs_server|dashboard.py|dashboard/agent.py|log_monitor.py' cpu)
  matched_total_rss=$(process_metric 'RLinf_ogpo_pi0_robotwin|ray::|raylet|gcs_server' rss)
  compute_count=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF {n += 1} END {print n + 0}')
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$now" "$host_available" "$cgroup_current" "$cgroup_anon" "$cgroup_file" \
    "$cgroup_oom" "$cgroup_oom_kill" "$shm_used" "$disk_available" \
    "${gpu0:--1,-1,-1,-1,-1}" "${gpu1:--1,-1,-1,-1,-1}" \
    "$env_rss" "$env_cpu" "$actor_rss" "$actor_cpu" "$rollout_rss" "$rollout_cpu" \
    "$driver_rss" "$driver_cpu" "$ray_rss" "$ray_cpu" "$matched_total_rss" "$compute_count" \
    >>"$output_csv"
}

while kill -0 "$driver_pid" 2>/dev/null; do
  sample_once
  sleep "$interval_seconds"
done
sample_once
