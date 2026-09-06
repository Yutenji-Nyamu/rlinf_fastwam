#!/usr/bin/env bash
set -u

driver_pid=$1
output_csv=$2
interval_seconds=${3:-2}

printf '%s\n' \
  'unix_time,cgroup_current_bytes,cgroup_anon_bytes,cgroup_file_bytes,cgroup_oom_events,cgroup_oom_kill_events,disk_available_bytes,gpu0_used_mib,gpu0_util_pct,gpu1_used_mib,gpu1_util_pct,driver_rss_kib,qam_process_rss_kib,compute_process_count' \
  >"${output_csv}"

memory_event() {
  awk -v key="$1" '$1 == key {print $2}' \
    /sys/fs/cgroup/memory.events 2>/dev/null || printf '%s' -1
}

sample_once() {
  now=$(date +%s)
  cgroup_current=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || printf '%s' -1)
  cgroup_anon=$(
    awk '$1 == "anon" {print $2}' /sys/fs/cgroup/memory.stat 2>/dev/null \
      || printf '%s' -1
  )
  cgroup_file=$(
    awk '$1 == "file" {print $2}' /sys/fs/cgroup/memory.stat 2>/dev/null \
      || printf '%s' -1
  )
  disk_available=$(
    df -B1 --output=avail /root/autodl-tmp 2>/dev/null | tail -n 1
  )
  gpu_rows=$(
    nvidia-smi \
      --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null || true
  )
  gpu0_used=$(printf '%s\n' "${gpu_rows}" | awk -F, '$1 + 0 == 0 {gsub(/ /, "", $2); print $2}')
  gpu0_util=$(printf '%s\n' "${gpu_rows}" | awk -F, '$1 + 0 == 0 {gsub(/ /, "", $3); print $3}')
  gpu1_used=$(printf '%s\n' "${gpu_rows}" | awk -F, '$1 + 0 == 1 {gsub(/ /, "", $2); print $2}')
  gpu1_util=$(printf '%s\n' "${gpu_rows}" | awk -F, '$1 + 0 == 1 {gsub(/ /, "", $3); print $3}')
  driver_rss=$(
    ps -o rss= -p "${driver_pid}" 2>/dev/null \
      | awk 'NF {print $1; found=1} END {if (!found) print 0}'
  )
  qam_rss=$(
    ps -eo rss=,args= \
      | awk \
        '$0 ~ /RLinf_qam_pi0_robotwin|QAMFSDPPolicy|MultiStepRolloutWorker|EnvWorker/ &&
         $0 !~ /qam_resource_monitor|awk/ {total += $1} END {print total + 0}'
  )
  compute_count=$(
    nvidia-smi \
      --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null \
      | awk 'NF {count += 1} END {print count + 0}'
  )
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${now}" \
    "${cgroup_current}" \
    "${cgroup_anon}" \
    "${cgroup_file}" \
    "$(memory_event oom)" \
    "$(memory_event oom_kill)" \
    "${disk_available// /}" \
    "${gpu0_used:--1}" \
    "${gpu0_util:--1}" \
    "${gpu1_used:--1}" \
    "${gpu1_util:--1}" \
    "${driver_rss}" \
    "${qam_rss}" \
    "${compute_count}" \
    >>"${output_csv}"
}

while kill -0 "${driver_pid}" 2>/dev/null; do
  sample_once
  sleep "${interval_seconds}"
done
sample_once
