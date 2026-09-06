#!/usr/bin/env bash
set -uo pipefail

run_dir='/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-formal100-4gpu128train64eval-official-v1'
metrics="$run_dir/metrics.log"
driver="$run_dir/driver.log"

date --iso-8601=seconds
hostname
id
printf 'run_dir=%s\n' "$run_dir"
stat -c 'metrics size=%s mtime=%y path=%n' "$metrics"
stat -c 'driver size=%s mtime=%y path=%n' "$driver"

printf 'latest_global_steps\n'
grep -a 'Global Step:' "$metrics" | tail -n 5 || true

printf 'step47_metric_block\n'
grep -a -A55 -B2 'Global Step:   47/100' "$metrics" | tail -n 60 || true

printf 'driver_tail\n'
tail -n 120 "$driver"

printf 'owned_run_processes\n'
for pid in 1375834 1376007 1376207 1376715 1383900 1383901 1383902 1383904 1383905 1383906 1383908 1383911 1383912 1384178 1384179 1384181; do
  if [[ -d "/proc/$pid" ]]; then
    ps -o user=,pid=,ppid=,stat=,comm= -p "$pid"
  fi
done
ps -u chenyiteng -o user=,pid=,ppid=,stat=,comm= | awk '$5 ~ /^(ray::|gcs_server|raylet)$/ {print}'

printf 'gpu_compute_processes\n'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader || true

printf 'checkpoint_directories\n'
find "$run_dir" -mindepth 1 -maxdepth 3 -type d -name 'global_step_*' -printf '%P\n' 2>/dev/null | sort -V || true

printf 'terminal_probe_complete\n'
