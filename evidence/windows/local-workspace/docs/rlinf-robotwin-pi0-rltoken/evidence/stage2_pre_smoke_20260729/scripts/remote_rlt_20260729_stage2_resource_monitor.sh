#!/usr/bin/env bash
set -u

driver_pid=$1
output_csv=$2
interval_seconds=${3:-2}

printf '%s\n' \
  'unix_time,host_available_bytes,cgroup_current_bytes,cgroup_anon_bytes,cgroup_file_bytes,cgroup_high_events,cgroup_max_events,cgroup_oom_events,cgroup_oom_kill_events,shm_used_bytes,disk_available_bytes,gpu0_used_mib,gpu0_util_pct,gpu1_used_mib,gpu1_util_pct,env_rss_kib,actor_rss_kib,rollout_rss_kib,driver_rss_kib,ray_system_rss_kib,matched_total_rss_kib,compute_process_count' \
  >"${output_csv}"

memory_event() {
  key=$1
  awk -v key="${key}" '$1 == key {print $2}' \
    /sys/fs/cgroup/memory.events 2>/dev/null \
    || printf '%s' -1
}

process_rss() {
  include_pattern=$1
  exclude_pattern=${2:-'^$'}
  ps -eo rss=,args= | awk \
    -v include="${include_pattern}" \
    -v exclude="${exclude_pattern}" \
    '$0 ~ include && $0 !~ exclude {total += $1} END {print total + 0}'
}

sample_once() {
  now=$(date +%s)
  host_available=$(awk '/MemAvailable:/ {print $2 * 1024}' /proc/meminfo)
  cgroup_current=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || printf '%s' -1)
  cgroup_anon=$(
    awk '$1 == "anon" {print $2}' \
      /sys/fs/cgroup/memory.stat 2>/dev/null \
      || printf '%s' -1
  )
  cgroup_file=$(
    awk '$1 == "file" {print $2}' \
      /sys/fs/cgroup/memory.stat 2>/dev/null \
      || printf '%s' -1
  )
  cgroup_high=$(memory_event high)
  cgroup_max=$(memory_event max)
  cgroup_oom=$(memory_event oom)
  cgroup_oom_kill=$(memory_event oom_kill)
  shm_used=$(df -B1 --output=used /dev/shm 2>/dev/null | tail -n 1)
  disk_available=$(
    df -B1 --output=avail /root/autodl-tmp 2>/dev/null | tail -n 1
  )

  gpu_rows=$(
    nvidia-smi \
      --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null \
      || true
  )
  gpu0_used=$(printf '%s\n' "${gpu_rows}" | awk -F, '$1 + 0 == 0 {gsub(/ /, "", $2); print $2}')
  gpu0_util=$(printf '%s\n' "${gpu_rows}" | awk -F, '$1 + 0 == 0 {gsub(/ /, "", $3); print $3}')
  gpu1_used=$(printf '%s\n' "${gpu_rows}" | awk -F, '$1 + 0 == 1 {gsub(/ /, "", $2); print $2}')
  gpu1_util=$(printf '%s\n' "${gpu_rows}" | awk -F, '$1 + 0 == 1 {gsub(/ /, "", $3); print $3}')

  env_rss=$(process_rss 'EnvWorker')
  actor_rss=$(process_rss 'EmbodiedFSDPActor|EmbodiedSACFSDPPolicy')
  rollout_rss=$(process_rss 'MultiStepRolloutWorker')
  driver_rss=$(
    ps -o rss= -p "${driver_pid}" 2>/dev/null \
      | awk 'NF {print $1; found=1} END {if (!found) print 0}'
  )
  ray_system_rss=$(
    process_rss \
      'raylet|gcs_server|dashboard.py|dashboard/agent.py|log_monitor.py|runtime_env/agent/main.py'
  )
  matched_total_rss=$(
    process_rss \
      'RLinf_rlt_pi0_robotwin|ray::|raylet|gcs_server' \
      'rlt_stage2_resource_monitor|awk -v include='
  )
  compute_count=$(
    nvidia-smi \
      --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null \
      | awk 'NF {count += 1} END {print count + 0}'
  )

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${now}" \
    "${host_available}" \
    "${cgroup_current}" \
    "${cgroup_anon}" \
    "${cgroup_file}" \
    "${cgroup_high}" \
    "${cgroup_max}" \
    "${cgroup_oom}" \
    "${cgroup_oom_kill}" \
    "${shm_used// /}" \
    "${disk_available// /}" \
    "${gpu0_used:--1}" \
    "${gpu0_util:--1}" \
    "${gpu1_used:--1}" \
    "${gpu1_util:--1}" \
    "${env_rss}" \
    "${actor_rss}" \
    "${rollout_rss}" \
    "${driver_rss}" \
    "${ray_system_rss}" \
    "${matched_total_rss}" \
    "${compute_count}" \
    >>"${output_csv}"
}

while kill -0 "${driver_pid}" 2>/dev/null; do
  sample_once
  sleep "${interval_seconds}"
done
sample_once
