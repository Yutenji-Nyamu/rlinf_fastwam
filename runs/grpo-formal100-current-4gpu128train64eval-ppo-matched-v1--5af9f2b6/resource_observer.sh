#!/usr/bin/env bash
set -u
pid=$1
out=$2
printf '%s\n' 'timestamp,driver_alive,host_mem_available_kib,cgroup_memory_current_bytes,gpu4_used_mib,gpu4_util_pct,gpu5_used_mib,gpu5_util_pct,gpu6_used_mib,gpu6_util_pct,gpu7_used_mib,gpu7_util_pct' > "$out"
while kill -0 "$pid" 2>/dev/null; do
  ts=$(date --iso-8601=seconds)
  mem=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  rel=$(awk -F: '$1=="0" {print $3}' "/proc/$pid/cgroup" 2>/dev/null)
  current=''
  test -n "$rel" && test -r "/sys/fs/cgroup$rel/memory.current" && current=$(cat "/sys/fs/cgroup$rel/memory.current")
  mapfile -t gpu < <(nvidia-smi -i 4,5,6,7 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  printf '%s,1,%s,%s' "$ts" "$mem" "$current" >> "$out"
  for row in "${gpu[@]}"; do printf ',%s' "$row" >> "$out"; done
  printf '\n' >> "$out"
  sleep 60
done
printf '%s,0,%s\n' "$(date --iso-8601=seconds)" "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" >> "$out"
