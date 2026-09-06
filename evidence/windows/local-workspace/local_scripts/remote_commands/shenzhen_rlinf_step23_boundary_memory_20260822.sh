#!/usr/bin/env bash
set -u

RUN=/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-formal100-4gpu128train64eval-official-v1

echo '=== TIME ==='
date --iso-8601=seconds
echo '=== PROGRESS ==='
grep -aE 'Global Step|Generating Rollout Epochs:' "$RUN/driver.log" | tail -n 18

driver_pid=$(cat "$RUN/driver.pid" 2>/dev/null || true)
if test -n "$driver_pid" && test -r "/proc/$driver_pid/cgroup"; then
  cgrel=$(awk -F: '$1=="0" {print $3}' "/proc/$driver_pid/cgroup")
  cg="/sys/fs/cgroup$cgrel"
  echo '=== CGROUP ==='
  printf 'memory.current '; cat "$cg/memory.current"
  if test -r "$cg/memory.peak"; then printf 'memory.peak '; cat "$cg/memory.peak"; fi
  cat "$cg/memory.events"
  awk '/^(anon|file|shmem|inactive_anon|active_anon|inactive_file|active_file) / {print}' "$cg/memory.stat"
fi

echo '=== HOST ==='
awk '/^(MemAvailable|AnonPages|Cached|Shmem|SwapTotal|SwapFree):/ {print}' /proc/meminfo

echo '=== ENVWORKERS ==='
for pid in $(ps -u "$(id -u)" -o pid=,comm= | awk '$2=="ray::EnvWorker" {print $1}'); do
  printf 'pid=%s ' "$pid"
  awk '
    /^Rss:/ {rss=$2}
    /^Pss:/ {pss=$2}
    /^Pss_Anon:/ {anon=$2}
    /^Pss_File:/ {file=$2}
    /^Pss_Shmem:/ {shmem=$2}
    /^Private_Dirty:/ {dirty=$2}
    END {printf "rss_kib=%s pss_kib=%s pss_anon_kib=%s pss_file_kib=%s pss_shmem_kib=%s private_dirty_kib=%s\n",rss,pss,anon,file,shmem,dirty}
  ' "/proc/$pid/smaps_rollup"
done

echo 'STEP23_BOUNDARY_MEMORY_DONE'
