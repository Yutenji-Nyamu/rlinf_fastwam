#!/usr/bin/env bash
set -uo pipefail

run='/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-formal100-current-4gpu128train64eval-ppo-matched-v1'
driver_pid="$(cat "$run/driver.pid" 2>/dev/null || true)"
observer_pid="$(cat "$run/resource_observer.pid" 2>/dev/null || true)"

date --iso-8601=seconds
hostname
id
printf 'run=%s\n' "$run"
printf 'driver_pid=%s observer_pid=%s\n' "$driver_pid" "$observer_pid"
for pid in "$driver_pid" "$observer_pid"; do
  [[ -n "$pid" ]] || continue
  if [[ -d "/proc/$pid" ]]; then
    ps -o user=,pid=,ppid=,etimes=,rss=,stat=,comm= -p "$pid"
  else
    printf 'pid=%s alive=false\n' "$pid"
  fi
done

printf 'resolved_and_outputs\n'
stat -c 'size=%s mtime=%y path=%n' "$run/resolved.yaml" "$run/driver.log" "$run/resource.csv" 2>/dev/null || true
cat "$run/resolved.yaml.sha256" 2>/dev/null || true
find "$run" -maxdepth 3 -type f -printf '%s %p\n' 2>/dev/null | sort -n | tail -n 40

printf 'ray_workers\n'
ps -u chenyiteng -o user=,pid=,ppid=,etimes=,rss=,stat=,comm= | awk '$7 ~ /^(ray::|gcs_server|raylet)/ {print; count[$7]++} END {for(c in count) print "count",c,count[c]}'

printf 'driver_stage_tail\n'
tail -n 160 "$run/driver.log" 2>/dev/null || true
printf 'fatal_matches\n'
grep -aEin 'traceback|cuda out of memory|illegal instruction|raytaskerror|segmentation fault|fatal|nccl.*error|exception|error:' "$run/driver.log" 2>/dev/null | tail -n 60 || true

printf 'resource_tail\n'
tail -n 8 "$run/resource.csv" 2>/dev/null || true

printf 'gpu_all\n'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
printf 'gpu_compute_processes\n'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader || true

printf 'memory_now\n'
free -h
awk '/MemTotal|MemAvailable|SwapTotal|SwapFree/ {print}' /proc/meminfo
if [[ -n "$driver_pid" && -r "/proc/$driver_pid/cgroup" ]]; then
  rel="$(awk -F: '$1=="0" {print $3}' "/proc/$driver_pid/cgroup")"
  printf 'driver_cgroup=%s\n' "$rel"
  [[ -r "/sys/fs/cgroup$rel/memory.current" ]] && cat "/sys/fs/cgroup$rel/memory.current"
  [[ -r "/sys/fs/cgroup$rel/memory.events" ]] && cat "/sys/fs/cgroup$rel/memory.events"
fi

printf 'fastwam_status\n'
if [[ -d /proc/637492 ]]; then
  ps -o user=,pid=,ppid=,etimes=,rss=,stat=,comm= -p 637492
else
  printf 'fastwam_pid_alive=false\n'
fi
tail -n 20 /data/chenyiteng/results/dvac-observation/run-metadata/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2/driver.log 2>/dev/null || true

printf 'startup_readonly_done\n'
