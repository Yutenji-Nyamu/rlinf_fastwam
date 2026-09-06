#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

runtime=/root/autodl-tmp/experiment_exports/rlt_stage2_formal_8env_250c_20260730_v1/runtime
run=/root/autodl-tmp/experiments/rlt_stage2_formal_8env_250c_20260730_v1/robotwin_adjust_bottle_rlt_stage2_formal_8env_250c_v1
driver="$(cat "${runtime}/driver_pid.txt")"

printf 'observed_at\t%s\n' "$(date -Iseconds)"
printf 'driver_pid\t%s\n' "${driver}"
printf 'driver_process\n'
ps -p "${driver}" -o pid=,stat=,etime=,%cpu=,%mem=,rss=,cmd= || true
printf 'started_at\t'
if test -f "${runtime}/started_at.txt"; then
  cat "${runtime}/started_at.txt"
else
  printf 'absent\n'
fi
printf 'finished_at\t'
if test -f "${runtime}/finished_at.txt"; then
  cat "${runtime}/finished_at.txt"
else
  printf 'absent\n'
fi
printf 'exit_code\t'
if test -f "${runtime}/exit_code.txt"; then
  cat "${runtime}/exit_code.txt"
else
  printf 'absent\n'
fi
printf 'latest_global_step\n'
grep -a 'Global Step:' "${runtime}/driver.log" | tail -n 1 || true
printf 'latest_success\n'
grep -a 'env/success_once' "${runtime}/driver.log" | tail -n 2 || true
printf 'latest_eval\n'
grep -a 'eval/success_once' "${runtime}/driver.log" | tail -n 3 || true
printf 'latest_update\n'
grep -aE 'update_step|pending_update_budget|actor/grad_norm|critic/grad_norm|actor_loss|critic_loss' \
  "${runtime}/driver.log" | tail -n 12 || true
printf 'fatal_scan\n'
grep -aEi 'CUDA out of memory|OutOfMemoryError|NCCL.*(error|fatal)|ray.*(died|death)|(^|[^[:alpha:]])nan([^[:alpha:]]|$)|(^|[^[:alpha:]])inf([^[:alpha:]]|$)' \
  "${runtime}/driver.log" | tail -n 20 || true
printf 'gpu\n'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
printf 'cgroup_current\t%s\n' "$(cat /sys/fs/cgroup/memory.current)"
grep -E '^(anon|file|kernel|shmem) ' /sys/fs/cgroup/memory.stat
printf 'memory_events\n'
cat /sys/fs/cgroup/memory.events
printf 'checkpoints\n'
find "${run}/checkpoints" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
  | sort -V
printf 'checkpoint_sizes\n'
du -sh "${run}/checkpoints"/global_step_* 2>/dev/null | sort -V
printf 'disk\n'
df -h /root/autodl-tmp
