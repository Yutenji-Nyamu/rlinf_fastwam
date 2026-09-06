#!/usr/bin/env bash
set -uo pipefail

run_dir='/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-formal100-4gpu128train64eval-official-v1'

date --iso-8601=seconds
hostname
id

printf 'known_owned_pid_survivors\n'
survivors=0
for pid in 1375834 1376007 1376207 1376715 1383900 1383901 1383902 1383904 1383905 1383906 1383908 1383911 1383912 1384178 1384179 1384181; do
  if [[ -d "/proc/$pid" ]]; then
    ps -o user=,pid=,ppid=,stat=,comm= -p "$pid"
    survivors=$((survivors + 1))
  fi
done
printf 'known_owned_pid_survivor_count=%d\n' "$survivors"

printf 'chenyiteng_ray_core_processes\n'
ps -u chenyiteng -o user=,pid=,ppid=,stat=,comm= | awk '$5 ~ /^(ray::|gcs_server|raylet)$/ {print; count++} END {printf "ray_core_count=%d\n",count+0}'

printf 'gpu_compute_processes\n'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader || true

printf 'checkpoint_directories\n'
find "$run_dir" -mindepth 1 -maxdepth 3 -type d -name 'global_step_*' -printf '%P\n' 2>/dev/null | sort -V || true

printf 'terminal_process_check_complete\n'
