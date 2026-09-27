#!/usr/bin/env bash
set -u
pid=$1; out=$2; gpu_a=$3; gpu_b=$4
printf '%s\n' "timestamp,driver_alive,host_mem_available_kib,gpu${gpu_a}_used_mib,gpu${gpu_a}_util_pct,gpu${gpu_b}_used_mib,gpu${gpu_b}_util_pct" > "$out"
while kill -0 "$pid" 2>/dev/null; do
  printf '%s,1,%s' "$(date --iso-8601=seconds)" "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" >> "$out"
  while IFS= read -r row; do printf ',%s' "$(tr -d ' ' <<< "$row")" >> "$out"; done < <(nvidia-smi -i "$gpu_a,$gpu_b" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  printf '\n' >> "$out"
  sleep 60
done
printf '%s,0,%s\n' "$(date --iso-8601=seconds)" "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" >> "$out"
