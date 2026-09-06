#!/usr/bin/env bash
set -u

RUN=/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-formal100-4gpu128train64eval-official-v1
ROOT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-pi0-robotwin
OFFICIAL="$ROOT/examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi.yaml"

echo '=== TIME ==='
date --iso-8601=seconds

echo '=== CURRENT_PROGRESS ==='
if test -f "$RUN/driver.log"; then
  grep -aE 'Global Step|Generating Rollout Epochs:' "$RUN/driver.log" | tail -n 16
fi

echo '=== SOURCE_LOCKS ==='
git -C "$ROOT" rev-parse HEAD
sha256sum "$OFFICIAL" "$RUN/resolved.yaml" "$RUN/launch_manifest.txt"

echo '=== OFFICIAL_YAML ==='
sed -n '1,240p' "$OFFICIAL"

echo '=== ACTUAL_RESOLVED ==='
sed -n '1,260p' "$RUN/resolved.yaml"

echo '=== LAUNCH_MANIFEST ==='
cat "$RUN/launch_manifest.txt"

echo '=== HOST_MEMINFO_KIB ==='
awk '/^(MemTotal|MemFree|MemAvailable|Buffers|Cached|SwapCached|Active|Inactive|Active\(anon\)|Inactive\(anon\)|Active\(file\)|Inactive\(file\)|Unevictable|Mlocked|SwapTotal|SwapFree|Dirty|Writeback|AnonPages|Mapped|Shmem|KReclaimable|Slab|SReclaimable|SUnreclaim|KernelStack|PageTables|Committed_AS|AnonHugePages|ShmemHugePages|FileHugePages):/ {print}' /proc/meminfo

driver_pid=$(cat "$RUN/driver.pid" 2>/dev/null || true)
echo '=== DRIVER_CGROUP ==='
if test -n "$driver_pid" && test -r "/proc/$driver_pid/cgroup"; then
  cat "/proc/$driver_pid/cgroup"
  cgrel=$(awk -F: '$1=="0" {print $3}' "/proc/$driver_pid/cgroup")
  cgroot=/sys/fs/cgroup
  cg="$cgroot$cgrel"
  echo "cgroup_path=$cg"
  for f in memory.current memory.peak memory.min memory.low memory.high memory.max memory.swap.current memory.swap.max memory.events memory.events.local memory.stat; do
    if test -r "$cg/$f"; then
      echo "--- $f ---"
      cat "$cg/$f"
    fi
  done
fi

echo '=== PROCESS_ROLES_RSS_KIB ==='
ps -u "$(id -u)" -o pid=,ppid=,rss=,vsz=,etimes=,comm=,args= | awk '
  function role(comm,args) {
    if (comm == "ray::EnvWorker") return "EnvWorker";
    if (comm ~ /ActorWorker/ || args ~ /EmbodiedFSDPActor/) return "ActorWorker";
    if (comm ~ /RolloutWorker/ || args ~ /MultiStepRolloutWorker/) return "RolloutWorker";
    if (comm == "raylet" || comm == "gcs_server" || args ~ /raylet|gcs_server|dashboard.py/) return "RayCore";
    if (args ~ /train_embodied_agent.py/) return "Driver";
    return "Other";
  }
  {
    r=role($6,$0); rss[r]+=$3; n[r]++;
  }
  END {for (r in rss) printf "%s count=%d rss_kib=%d\n", r,n[r],rss[r]}
' | sort

echo '=== PPO_PROCESS_TABLE ==='
ps -u "$(id -u)" -o pid=,ppid=,rss=,vsz=,etimes=,comm=,args= \
  | grep -E 'ray::EnvWorker|ActorWorker|RolloutWorker|EmbodiedFSDPActor|MultiStepRolloutWorker|train_embodied_agent.py|raylet|gcs_server' \
  | grep -v grep \
  | sort -k3,3nr

echo '=== ENVWORKER_STATUS_AND_SMAPS_ROLLUP_KIB ==='
for pid in $(ps -u "$(id -u)" -o pid=,comm= | awk '$2=="ray::EnvWorker" {print $1}'); do
  echo "--- PID $pid STATUS ---"
  awk '/^(Name|Pid|PPid|Threads|VmPeak|VmSize|VmHWM|VmRSS|RssAnon|RssFile|RssShmem|VmData|VmSwap):/ {print}' "/proc/$pid/status"
  echo "--- PID $pid SMAPS_ROLLUP ---"
  awk '/^(Rss|Pss|Pss_Anon|Pss_File|Pss_Shmem|Shared_Clean|Shared_Dirty|Private_Clean|Private_Dirty|Referenced|Anonymous|LazyFree|AnonHugePages|ShmemPmdMapped|FilePmdMapped|Shared_Hugetlb|Private_Hugetlb|Swap|SwapPss|Locked):/ {print}' "/proc/$pid/smaps_rollup" 2>&1 || true
done

echo '=== PROC_VMSTAT_SELECTED ==='
awk '/^(pgfault|pgmajfault|pswpin|pswpout|compact_stall|compact_fail|compact_success|oom_kill) / {print}' /proc/vmstat

echo 'FORMAL_CONFIG_MEMORY_DEEPREAD_DONE'
