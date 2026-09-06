#!/usr/bin/env bash
set -u
RUN=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822
RUNTIME=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822

date --iso-8601=seconds
hostname
printf '%s\n' '=== OWNED PROCESSES ==='
for name in wrapper driver observer; do
  pid=$(cat "$RUNTIME/$name.pid" 2>/dev/null || true)
  if [ -n "$pid" ] && [ -d "/proc/$pid" ]; then
    ps -p "$pid" -o pid=,etimes=,rss=,args=
  else
    printf '%s\t%s\n' "$name" dead_or_missing
  fi
done
printf '%s\n' '=== LATEST COMPLETE STEPS ==='
grep -a 'Global Step' "$RUN/metrics.log" 2>/dev/null | tail -n 4 || true
printf '%s\n' '=== CURRENT CHECKPOINTS ==='
find "$RUN/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -printf '%f\t%TY-%Tm-%TdT%TH:%TM:%TS\n' 2>/dev/null | sort -V
printf '%s\n' '=== GPU ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
printf '%s\n' '=== CGROUP ==='
printf 'current='; cat /sys/fs/cgroup/memory.current
printf 'peak='; cat /sys/fs/cgroup/memory.peak
printf 'max='; cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/memory.events
printf '%s\n' '=== DISK ==='
df -h /root/autodl-tmp
du -sh "$RUN" "$RUNTIME" 2>/dev/null
printf '%s\n' '=== ERROR TAIL ==='
grep -aE 'CUDA out of memory|NCCL|WorkerCrashed|RayTaskError|No space left|SIGSEGV|SIGBUS|Killed|all_gather.*(error|Error)' "$RUNTIME/driver.log" 2>/dev/null | tail -n 20 || true
