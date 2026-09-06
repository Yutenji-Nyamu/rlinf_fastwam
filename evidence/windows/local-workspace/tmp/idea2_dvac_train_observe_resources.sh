#!/usr/bin/env bash
set -u

driver_pid=$1
monitor_dir=$2
target_tag=/root/autodl-tmp/RLinf_idea2_dvac_train
mkdir -p "$monitor_dir"

resources_csv="$monitor_dir/resources.csv"
process_tsv="$monitor_dir/process_rss.tsv"
printf '%s\n' 'timestamp,elapsed_s,gpu_index,gpu_memory_used_mib,gpu_memory_total_mib,gpu_util_pct,gpu_memory_util_pct,temperature_c,power_w,cgroup_current_bytes,cgroup_anon_bytes,cgroup_file_bytes,event_low,event_high,event_max,event_oom,event_oom_kill,host_mem_available_kib,shm_available_kib,disk_available_kib,gpu_compute_process_count,driver_alive' > "$resources_csv"
printf 'timestamp\tpid\tppid\trss_kib\tpcpu\tcomm\targs\n' > "$process_tsv"

start_epoch=$(date +%s)
sample_index=0
while test -d "/proc/$driver_pid"; do
  timestamp=$(date --iso-8601=seconds)
  now_epoch=$(date +%s)
  elapsed_s=$((now_epoch - start_epoch))
  cgroup_current=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || printf '')
  cgroup_anon=$(awk '$1 == "anon" {print $2}' /sys/fs/cgroup/memory.stat 2>/dev/null)
  cgroup_file=$(awk '$1 == "file" {print $2}' /sys/fs/cgroup/memory.stat 2>/dev/null)
  event_low=$(awk '$1 == "low" {print $2}' /sys/fs/cgroup/memory.events 2>/dev/null)
  event_high=$(awk '$1 == "high" {print $2}' /sys/fs/cgroup/memory.events 2>/dev/null)
  event_max=$(awk '$1 == "max" {print $2}' /sys/fs/cgroup/memory.events 2>/dev/null)
  event_oom=$(awk '$1 == "oom" {print $2}' /sys/fs/cgroup/memory.events 2>/dev/null)
  event_oom_kill=$(awk '$1 == "oom_kill" {print $2}' /sys/fs/cgroup/memory.events 2>/dev/null)
  host_mem_available=$(awk '$1 == "MemAvailable:" {print $2}' /proc/meminfo 2>/dev/null)
  shm_available=$(df -Pk /dev/shm 2>/dev/null | awk 'NR == 2 {print $4}')
  disk_available=$(df -Pk /root/autodl-tmp 2>/dev/null | awk 'NR == 2 {print $4}')
  compute_count=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' | wc -l)

  nvidia-smi \
    --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw \
    --format=csv,noheader,nounits 2>/dev/null | sed 's/, /,/g' | \
    while IFS=',' read -r gpu_index memory_used memory_total gpu_util memory_util temperature power_draw; do
      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,1\n' \
        "$timestamp" "$elapsed_s" "$gpu_index" "$memory_used" "$memory_total" "$gpu_util" \
        "$memory_util" "$temperature" "$power_draw" "$cgroup_current" "$cgroup_anon" "$cgroup_file" \
        "$event_low" "$event_high" "$event_max" "$event_oom" "$event_oom_kill" \
        "$host_mem_available" "$shm_available" "$disk_available" "$compute_count"
    done >> "$resources_csv"

  if test $((sample_index % 5)) -eq 0; then
    ps -eo pid=,ppid=,rss=,pcpu=,comm=,args= | \
      awk -v ts="$timestamp" -v tag="$target_tag" \
        '$5 != "awk" && (index($0, tag) || $5 == "raylet" || index($0, "ray::")) {
          printf "%s\t%s\t%s\t%s\t%s\t%s\t", ts, $1, $2, $3, $4, $5;
          for (i = 6; i <= NF; ++i) printf "%s%s", $i, (i == NF ? ORS : OFS)
        }' >> "$process_tsv"
  fi

  sample_index=$((sample_index + 1))
  sleep 2
done

printf 'OBSERVER_NATURAL_EXIT_AT=%s\nSAMPLES=%s\n' \
  "$(date --iso-8601=seconds)" "$sample_index" > "$monitor_dir/observer_exit.txt"
