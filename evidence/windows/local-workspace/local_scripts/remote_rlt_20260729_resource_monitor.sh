set -u

DRIVER_PID=$1
OUTPUT_CSV=$2

printf '%s\n' \
  'unix_time,host_available_bytes,cgroup_current_bytes,cgroup_anon_bytes,cgroup_file_bytes,gpu0_used_mib,gpu0_util_pct,gpu1_used_mib,gpu1_util_pct,matched_rss_kib,compute_process_count' \
  > "$OUTPUT_CSV"

sample_once() {
  now=$(date +%s)
  host_available=$(awk '/MemAvailable:/ {print $2 * 1024}' /proc/meminfo)
  cgroup_current=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || printf '%s' -1)
  cgroup_anon=$(awk '$1 == "anon" {print $2}' /sys/fs/cgroup/memory.stat 2>/dev/null || printf '%s' -1)
  cgroup_file=$(awk '$1 == "file" {print $2}' /sys/fs/cgroup/memory.stat 2>/dev/null || printf '%s' -1)

  gpu_rows=$(nvidia-smi \
    --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null || true)
  gpu0_used=$(printf '%s\n' "$gpu_rows" | awk -F, '$1 + 0 == 0 {gsub(/ /, "", $2); print $2}')
  gpu0_util=$(printf '%s\n' "$gpu_rows" | awk -F, '$1 + 0 == 0 {gsub(/ /, "", $3); print $3}')
  gpu1_used=$(printf '%s\n' "$gpu_rows" | awk -F, '$1 + 0 == 1 {gsub(/ /, "", $2); print $2}')
  gpu1_util=$(printf '%s\n' "$gpu_rows" | awk -F, '$1 + 0 == 1 {gsub(/ /, "", $3); print $3}')

  matched_rss=$(ps -eo rss=,args= | awk '
    /RLinf_rlt_pi0_robotwin|ray::|raylet|gcs_server/ &&
    $0 !~ /rlt_stage1_resource_monitor/ {
      total += $1
    }
    END {print total + 0}
  ')
  compute_count=$(nvidia-smi \
    --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null | awk 'NF {count += 1} END {print count + 0}')

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$now" \
    "$host_available" \
    "$cgroup_current" \
    "$cgroup_anon" \
    "$cgroup_file" \
    "${gpu0_used:--1}" \
    "${gpu0_util:--1}" \
    "${gpu1_used:--1}" \
    "${gpu1_util:--1}" \
    "$matched_rss" \
    "$compute_count" \
    >> "$OUTPUT_CSV"
}

while kill -0 "$DRIVER_PID" 2>/dev/null; do
  sample_once
  sleep 1
done
sample_once
