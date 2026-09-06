#!/usr/bin/env bash
set -uo pipefail

pi0='/data/chenyiteng/results/dvac-observation/pi0-adjust_bottle-p1-16ep-800baf80-v1'
fastwam_meta='/data/chenyiteng/results/dvac-observation/run-metadata/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2'
fastwam_payload='/data/chenyiteng/results/dvac-observation/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2'
fastwam_official='/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711/evaluate_results/robotwin/robotwin_uncond_3cam_384/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2'

date --iso-8601=seconds
hostname
id

printf 'pi0_root_and_ray\n'
pi0_pid="$(cat "$pi0/driver.pid" 2>/dev/null || true)"
printf 'pi0_pid=%s\n' "$pi0_pid"
if [[ -n "$pi0_pid" && -d "/proc/$pi0_pid" ]]; then
  ps -o user=,pid=,ppid=,etimes=,rss=,stat=,comm= -p "$pi0_pid"
else
  printf 'pi0_pid_alive=false\n'
fi
ps -u chenyiteng -o user=,pid=,ppid=,etimes=,rss=,stat=,comm= | awk '$7 ~ /^(ray::|gcs_server|raylet)$/ {print; count++} END {printf "ray_core_count=%d\n",count+0}'

printf 'pi0_log_tail\n'
stat -c 'size=%s mtime=%y path=%n' "$pi0/driver.log" 2>/dev/null || true
tail -n 100 "$pi0/driver.log" 2>/dev/null || true
printf 'pi0_fatal_count\n'
grep -aEin 'traceback|cuda out of memory|illegal instruction|raytaskerror|segmentation fault|fatal|exception|error:' "$pi0/driver.log" 2>/dev/null | wc -l
printf 'pi0_artifact_counts\n'
find "$pi0" -type f 2>/dev/null | awk 'END {print "files=" NR}'
find "$pi0" -type f 2>/dev/null | sed 's#^.*/##' | awk -F. '{ext=(NF>1?$NF:"noext"); count[ext]++} END {for(e in count) print e,count[e]}' | sort
for pattern in 'queries.csv' 'episodes.csv' '*.npz' '*.png' '*.mp4' '*manifest*.json'; do
  count="$(find "$pi0" -type f -name "$pattern" 2>/dev/null | wc -l)"
  printf 'pi0_pattern=%s count=%s\n' "$pattern" "$count"
done
find "$pi0" -type f \( -name 'queries.csv' -o -name 'episodes.csv' -o -name '*manifest*.json' \) -printf '%s %p\n' 2>/dev/null | sort

printf 'fastwam_root\n'
printf 'requested_pid=637492\n'
if [[ -d /proc/637492 ]]; then
  ps -o user=,pid=,ppid=,etimes=,rss=,stat=,comm= -p 637492
  printf 'cmd='
  tr '\0' ' ' < /proc/637492/cmdline
  printf '\nstdout='
  readlink /proc/637492/fd/1 || true
  printf 'stderr='
  readlink /proc/637492/fd/2 || true
else
  printf 'fastwam_pid_alive=false\n'
fi

printf 'fastwam_log_tail\n'
for log in "$fastwam_meta/driver.log" "$fastwam_official"/*.log; do
  [[ -f "$log" ]] || continue
  stat -c 'size=%s mtime=%y path=%n' "$log"
  tail -n 120 "$log"
done
printf 'fastwam_fatal_count\n'
grep -aEihn 'traceback|cuda out of memory|illegal instruction|segmentation fault|fatal|exception|error:' "$fastwam_meta/driver.log" "$fastwam_official"/*.log 2>/dev/null | wc -l

printf 'fastwam_artifact_counts\n'
for root in "$fastwam_meta" "$fastwam_payload" "$fastwam_official"; do
  if [[ -e "$root" ]]; then
    printf 'root=%s\n' "$root"
    find "$root" -type f 2>/dev/null | awk 'END {print "files=" NR}'
    find "$root" -type f 2>/dev/null | sed 's#^.*/##' | awk -F. '{ext=(NF>1?$NF:"noext"); count[ext]++} END {for(e in count) print e,count[e]}' | sort
  else
    printf 'absent=%s\n' "$root"
  fi
done
for pattern in 'queries.csv' 'episodes.csv' '*.npz' '*.png' '*.mp4' '*manifest*.json'; do
  count="$(find "$fastwam_payload" "$fastwam_official" -type f -name "$pattern" 2>/dev/null | wc -l)"
  printf 'fastwam_pattern=%s count=%s\n' "$pattern" "$count"
done
find "$fastwam_payload" "$fastwam_official" -type f \( -name 'episodes.csv' -o -name 'queries.csv' -o -name '*.mp4' -o -name '*manifest*.json' \) -printf '%s %p\n' 2>/dev/null | sort

printf 'gpu_all\n'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
printf 'gpu_compute_processes\n'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader || true

printf 'memory_now\n'
free -h
awk '/MemTotal|MemAvailable|SwapTotal|SwapFree/ {print}' /proc/meminfo

printf 'completion_refresh_done\n'
