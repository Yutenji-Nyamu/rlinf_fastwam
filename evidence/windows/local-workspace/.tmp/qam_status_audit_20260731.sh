#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
run=/root/autodl-tmp/experiments/qam_formal_20260731_v1
runtime=/root/autodl-tmp/experiment_exports/qam_formal_20260731_v1/runtime

printf '=== IDENTITY ===\n'
hostname
pwd
id -u
date '+%F %T %Z'

printf '=== GIT ===\n'
git -C "$repo" branch --show-current
git -C "$repo" rev-parse HEAD
git -C "$repo" rev-parse '@{upstream}'
git -C "$repo" status --short
git -C "$repo" rev-list --left-right --count '@{upstream}...HEAD'

printf '=== PROCESS ===\n'
for pidfile in \
  /root/autodl-tmp/qam_formal_supervisor_20260731_v1.pid \
  "$runtime/driver.pid" \
  "$runtime/monitor.pid"
do
  if test -f "$pidfile"; then
    pid=$(cat "$pidfile")
    printf 'PIDFILE %s PID %s\n' "$pidfile" "$pid"
    ps -o pid,ppid,stat,lstart,etime,rss,cmd -p "$pid" || true
  else
    printf 'MISSING_PIDFILE %s\n' "$pidfile"
  fi
done
printf 'RAYLET_COUNT='; pgrep -xc raylet || true
printf 'GCS_COUNT='; pgrep -xc gcs_server || true
printf 'TRAIN_MATCHES\n'
pgrep -af '[t]rain_embodied_agent.py|[t]orch.distributed.run|ray::QAM|ray::MultiStepRolloutWorker|ray::EnvWorker' || true
for f in exit_code.txt monitor_exit_code.txt; do
  if test -f "$runtime/$f"; then printf '%s=' "$f"; cat "$runtime/$f"; else printf '%s=absent\n' "$f"; fi
done

printf '=== GPU ===\n'
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits

printf '=== MEMORY ===\n'
awk '/MemTotal|MemAvailable|Cached|SwapTotal|SwapFree/ {print}' /proc/meminfo
printf 'cgroup_current='; cat /sys/fs/cgroup/memory.current
printf 'cgroup_max='; cat /sys/fs/cgroup/memory.max
grep -E '^(anon|file|kernel|slab|pagetables|shmem) ' /sys/fs/cgroup/memory.stat
cat /sys/fs/cgroup/memory.events
cat /sys/fs/cgroup/memory.pressure

printf '=== DISK ===\n'
df -h /root/autodl-tmp
du -sh "$run" "$runtime" 2>/dev/null || true

printf '=== RUNTIME FILES ===\n'
find "$runtime" -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %12s %f\n' | sort
for f in driver.log resources.csv monitor.log; do
  if test -f "$runtime/$f"; then wc -l -c "$runtime/$f"; fi
done

printf '=== RESOURCE HEAD_TAIL ===\n'
head -n 3 "$runtime/resources.csv" 2>/dev/null || true
tail -n 5 "$runtime/resources.csv" 2>/dev/null || true

printf '=== TRAIN ARTIFACT SUMMARY ===\n'
printf 'video_files='; find "$run/video" -type f 2>/dev/null | wc -l
printf 'video_bytes='; find "$run/video" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}'
printf 'tb_files='; find "$run/tensorboard" -type f 2>/dev/null | wc -l
printf 'tb_bytes='; find "$run/tensorboard" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}'
printf 'checkpoint_dirs\n'
find "$run" -type d -path '*/checkpoints/global_step_*' -printf '%T@ %p\n' 2>/dev/null | sort -n
printf 'checkpoint_files_and_bytes='; find "$run" -path '*/checkpoints/global_step_*/*' -type f -printf '%s\n' 2>/dev/null | awk '{n+=1;s+=$1} END {print n+0, s+0}'

printf '=== DRIVER LATEST ===\n'
tail -n 220 "$runtime/driver.log" 2>/dev/null || true

printf '=== FATAL COUNTS ===\n'
for pattern in 'CUDA out of memory' 'OutOfMemory' 'NCCL' 'NaN' 'nan' 'SIGTERM' 'Killed' 'RayActorError'; do
  count=$(grep -c -F "$pattern" "$runtime/driver.log" 2>/dev/null || true)
  printf '%s\t%s\n' "$pattern" "$count"
done
