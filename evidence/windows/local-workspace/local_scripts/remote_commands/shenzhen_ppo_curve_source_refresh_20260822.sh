#!/usr/bin/env bash
set -uo pipefail
RUN=/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-formal100-4gpu128train64eval-official-v1

printf 'timestamp=%s\n' "$(TZ=Asia/Shanghai date --iso-8601=seconds)"
DRIVER=$(cat "$RUN/driver.pid" 2>/dev/null || true)
printf 'driver_pid=%s\n' "$DRIVER"
if test -n "$DRIVER" && kill -0 "$DRIVER" 2>/dev/null; then
  printf 'driver_alive=yes\n'
  ps -p "$DRIVER" -o lstart=,etime=,stat=,%cpu=,rss=,comm=
else
  printf 'driver_alive=no\n'
fi

printf '%s\n' '=== latest progress ==='
grep -aE 'Global Step:|Generating Rollout Epochs:|Evaluating Rollout Epochs:' "$RUN/driver.log" | tail -n 24 || true
printf 'fatal_count='; grep -aiEc 'Traceback|CUDA out of memory|OutOfMemory|WorkerCrashed|RayActorError|SIGKILL|Killed process|No space left|NCCL.*(error|failed)' "$RUN/driver.log" || true

printf '%s\n' '=== curve source files ==='
find "$RUN" -type f \( -name 'events.out.tfevents.*' -o -iname '*.csv' -o -iname '*resource*' -o -iname '*monitor*' -o -name 'metrics.log' \) \
  -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS\t%p\n' 2>/dev/null | sort -k3,3

printf '%s\n' '=== checkpoints ==='
find "$RUN/robotwin_ppo_openpi/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' 2>/dev/null | sort -V

printf '%s\n' '=== gpu ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>&1

printf '%s\n' '=== memory ==='
grep -E '^(MemTotal|MemAvailable|MemFree|Cached|SReclaimable|Shmem|SwapTotal|SwapFree):' /proc/meminfo
if test -n "$DRIVER" && test -r "/proc/$DRIVER/cgroup"; then
  CGREL=$(awk -F: '$1=="0" {print $3}' "/proc/$DRIVER/cgroup")
  CG=/sys/fs/cgroup${CGREL}
  printf 'driver_cgroup=%s\n' "$CG"
  for f in memory.current memory.peak memory.high memory.max memory.swap.current memory.events; do
    test -r "$CG/$f" && { printf -- '-- %s --\n' "$f"; cat "$CG/$f"; }
  done
fi
exit 0
