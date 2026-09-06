#!/usr/bin/env bash
set -uo pipefail

pi0='/data/chenyiteng/results/dvac-observation/pi0-adjust_bottle-p1-16ep-800baf80-v1'
fastwam='/data/chenyiteng/results/dvac-observation/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v1'
fastwam_eval='/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711/evaluate_results/robotwin/robotwin_uncond_3cam_384/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v1'

date --iso-8601=seconds
hostname
id

printf 'pi0_process_and_log\n'
cat "$pi0/driver.pid" 2>/dev/null || true
pi0_pid="$(cat "$pi0/driver.pid" 2>/dev/null || true)"
if [[ -n "$pi0_pid" && -d "/proc/$pi0_pid" ]]; then
  ps -o user=,pid=,ppid=,etimes=,rss=,stat=,comm= -p "$pi0_pid"
fi
stat -c 'size=%s mtime=%y path=%n' "$pi0/driver.log" 2>/dev/null || true
tail -n 80 "$pi0/driver.log" 2>/dev/null || true
printf 'pi0_fatal_matches\n'
grep -aEin 'traceback|cuda out of memory|illegal instruction|raytaskerror|segmentation fault|fatal|exception|error:' "$pi0/driver.log" 2>/dev/null | tail -n 40 || true
printf 'pi0_output_counts\n'
find "$pi0" -type f 2>/dev/null | awk 'END {print "files=" NR}'
find "$pi0" -type f 2>/dev/null | sed 's#^.*/##' | awk -F. '{ext=(NF>1?$NF:"noext"); count[ext]++} END {for(e in count) print e,count[e]}' | sort

printf 'fastwam_process_and_log\n'
cat "$fastwam/driver.pid" 2>/dev/null || true
fastwam_pid="$(cat "$fastwam/driver.pid" 2>/dev/null || true)"
if [[ -n "$fastwam_pid" && -d "/proc/$fastwam_pid" ]]; then
  ps -o user=,pid=,ppid=,etimes=,rss=,stat=,comm= -p "$fastwam_pid"
else
  printf 'fastwam_pid_alive=false\n'
fi
stat -c 'size=%s mtime=%y path=%n' "$fastwam/driver.log" 2>/dev/null || true
cat "$fastwam/driver.log" 2>/dev/null || true
printf 'fastwam_fatal_matches\n'
grep -aEin 'traceback|cuda out of memory|illegal instruction|segmentation fault|fatal|exception|error:' "$fastwam/driver.log" 2>/dev/null | tail -n 40 || true
printf 'fastwam_output_counts\n'
find "$fastwam" -type f 2>/dev/null | awk 'END {print "telemetry_files=" NR}'
find "$fastwam_eval" -type f 2>/dev/null | awk 'END {print "eval_files=" NR}'
find "$fastwam_eval" -type f -printf '%s %p\n' 2>/dev/null | sort -n | tail -n 40 || true

printf 'gpu_and_memory\n'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader || true
free -h

printf 'log_health_complete\n'
