#!/usr/bin/env bash
set -u

driver_pid=$1
output_csv=$2

printf '%s\n' \
  'timestamp,driver_alive,host_mem_available_kib,cgroup_memory_current_bytes,cgroup_oom,cgroup_oom_kill,mem_psi_some_avg10,mem_psi_full_avg10,gpu4_used_mib,gpu4_util_pct,gpu5_used_mib,gpu5_util_pct' \
  > "$output_csv"

while kill -0 "$driver_pid" 2>/dev/null; do
  timestamp=$(date --iso-8601=seconds)
  mem_available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  cgroup_rel=$(awk -F: '$1 == "0" {print $3}' "/proc/$driver_pid/cgroup" 2>/dev/null)
  cgroup_root="/sys/fs/cgroup$cgroup_rel"
  cgroup_current=$(cat "$cgroup_root/memory.current" 2>/dev/null || printf '')
  cgroup_oom=$(awk '$1 == "oom" {print $2}' "$cgroup_root/memory.events" 2>/dev/null || printf '')
  cgroup_oom_kill=$(awk '$1 == "oom_kill" {print $2}' "$cgroup_root/memory.events" 2>/dev/null || printf '')
  psi_some=$(awk '$1 == "some" {sub("avg10=", "", $2); print $2}' /proc/pressure/memory)
  psi_full=$(awk '$1 == "full" {sub("avg10=", "", $2); print $2}' /proc/pressure/memory)
  gpu4_row=$(nvidia-smi -i 4 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  gpu5_row=$(nvidia-smi -i 5 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')

  printf '%s,1,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$timestamp" \
    "$mem_available_kib" \
    "$cgroup_current" \
    "$cgroup_oom" \
    "$cgroup_oom_kill" \
    "$psi_some" \
    "$psi_full" \
    "$gpu4_row" \
    "$gpu5_row" \
    >> "$output_csv"
  sleep 10
done

printf '%s,0,%s,,,,,,,,,\n' \
  "$(date --iso-8601=seconds)" \
  "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" \
  >> "$output_csv"
