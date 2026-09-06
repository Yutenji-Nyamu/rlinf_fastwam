#!/usr/bin/env bash
set -uo pipefail

date --iso-8601=seconds
hostname
id

printf 'root_processes\n'
for pid in 619674 619300; do
  if [[ -d "/proc/$pid" ]]; then
    ps -o user=,pid=,ppid=,etimes=,rss=,stat=,comm= -p "$pid"
    printf 'pid=%s stdout=' "$pid"
    readlink "/proc/$pid/fd/1" || true
    printf 'pid=%s stderr=' "$pid"
    readlink "/proc/$pid/fd/2" || true
    printf 'pid=%s cmd=' "$pid"
    tr '\0' ' ' < "/proc/$pid/cmdline"
    printf '\n'
  else
    printf 'pid=%s absent\n' "$pid"
  fi
done

printf 'process_forest\n'
ps -u chenyiteng -o pid=,ppid=,etimes=,rss=,stat=,comm= --forest | grep -E '(^|[[:space:]])(619300|619674|timeout|eval_embodied|fastwam|ray::|gcs_server|raylet|python)([[:space:]]|$)' || true

printf 'gpu_summary\n'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
printf 'gpu_compute_processes\n'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader || true

printf 'memory_now\n'
free -h
awk '/MemAvailable|SwapTotal|SwapFree/ {print}' /proc/meminfo

printf 'recent_dvac_output_metadata\n'
find /data/chenyiteng/results/dvac-observation -maxdepth 3 -mmin -30 -printf '%TY-%Tm-%TdT%TH:%TM:%TS %y %s %p\n' 2>/dev/null | sort | tail -n 120 || true

printf 'recent_fastwam_result_metadata\n'
find /data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711/evaluate_results -maxdepth 6 -mmin -30 -printf '%TY-%Tm-%TdT%TH:%TM:%TS %y %s %p\n' 2>/dev/null | sort | tail -n 120 || true

printf 'health_probe_complete\n'
