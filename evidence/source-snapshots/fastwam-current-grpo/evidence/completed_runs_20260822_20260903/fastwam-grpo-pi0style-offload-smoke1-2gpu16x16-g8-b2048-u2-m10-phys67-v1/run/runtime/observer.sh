#!/usr/bin/env bash
set -u
pid=$1; out=$2
printf 'timestamp,mem_available_kib,gpu6_mem_mib,gpu6_util,gpu7_mem_mib,gpu7_util\n' > "$out"
while kill -0 "$pid" 2>/dev/null; do
  ts=$(date --iso-8601=seconds)
  mem=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  mapfile -t g < <(nvidia-smi -i 6,7 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  printf '%s,%s,%s,%s\n' "$ts" "$mem" "${g[0]}" "${g[1]}" >> "$out"
  sleep 2
done
