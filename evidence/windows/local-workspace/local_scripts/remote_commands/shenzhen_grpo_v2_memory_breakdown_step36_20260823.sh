#!/usr/bin/env bash
set -u

RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-formal100-current-4gpu128train64eval-ppo-matched-v2
DRIVER=$(cat "$RUN/driver.pid" 2>/dev/null || true)

printf 'TIME_IDENTITY\n'
date --iso-8601=seconds
id

printf 'OWNED_PROCESS_COUNTS\n'
printf 'actor='; pgrep -u "$(id -u)" -fc '^ray::EmbodiedFSDPActor' || true
printf 'rollout='; pgrep -u "$(id -u)" -fc '^ray::MultiStepRolloutWorker' || true
printf 'env='; pgrep -u "$(id -u)" -fc '^ray::EnvWorker' || true
printf 'gcs='; pgrep -u "$(id -u)" -xc gcs_server || true
printf 'raylet='; pgrep -u "$(id -u)" -xc raylet || true

printf 'TOP_RSS_KIB\n'
ps -u "$(id -u)" -o pid=,ppid=,rss=,vsz=,etimes=,stat=,comm= --sort=-rss | head -n 20

printf 'WORKER_PSS_KIB\n'
for pattern in '^ray::EnvWorker' '^ray::MultiStepRolloutWorker' '^ray::EmbodiedFSDPActor'; do
  for pid in $(pgrep -u "$(id -u)" -f "$pattern" || true); do
    name=$(cat "/proc/$pid/comm" 2>/dev/null || true)
    rss=$(awk '/^VmRSS:/ {print $2}' "/proc/$pid/status" 2>/dev/null || true)
    pss=$(awk '/^Pss:/ {print $2}' "/proc/$pid/smaps_rollup" 2>/dev/null || true)
    private_dirty=$(awk '/^Private_Dirty:/ {print $2}' "/proc/$pid/smaps_rollup" 2>/dev/null || true)
    printf 'pid=%s name=%s rss_kib=%s pss_kib=%s private_dirty_kib=%s\n' "$pid" "$name" "$rss" "$pss" "$private_dirty"
  done
done

printf 'CGROUP_MEMORY\n'
if test -n "$DRIVER" && test -r "/proc/$DRIVER/cgroup"; then
  rel=$(awk -F: '$1=="0" {print $3}' "/proc/$DRIVER/cgroup")
  cg="/sys/fs/cgroup$rel"
  printf 'cgroup=%s\n' "$cg"
  for name in memory.current memory.peak memory.swap.current memory.swap.peak; do
    test -r "$cg/$name" && printf '%s=%s\n' "$name" "$(cat "$cg/$name")"
  done
  cat "$cg/memory.events"
  printf 'memory_stat_selected\n'
  awk '$1 ~ /^(anon|file|kernel|kernel_stack|pagetables|percpu|sock|shmem|file_mapped|file_dirty|inactive_anon|active_anon|inactive_file|active_file|slab|slab_reclaimable|slab_unreclaimable)$/ {print}' "$cg/memory.stat"
  printf 'memory_pressure\n'
  cat "$cg/memory.pressure" 2>/dev/null || true
fi

printf 'HOST_VMSTAT_3S\n'
free -h
vmstat 1 3
printf 'host_memory_pressure\n'
cat /proc/pressure/memory 2>/dev/null || true

printf 'LATEST_RESOURCE_ROWS\n'
tail -n 5 "$RUN/resource.csv" 2>/dev/null || true
printf 'SZ_GRPO_V2_MEMORY_BREAKDOWN_STEP36_OK\n'

