#!/usr/bin/env bash
set -euo pipefail

control_run=/root/autodl-tmp/experiments/rlt_single_gpu_control_matched_width_formal480_20260826_v3
method_run=/root/autodl-tmp/experiments/rlt_single_gpu_success_episode_bc_dvac_matched_width_formal480_20260826_v3
control_rt=/root/autodl-tmp/experiment_exports/rlt_single_gpu_control_matched_width_formal480_20260826_v3/runtime
method_rt=/root/autodl-tmp/experiment_exports/rlt_single_gpu_success_episode_bc_dvac_matched_width_formal480_20260826_v3/runtime
pair_rt=/root/autodl-tmp/experiment_exports/rlt_success_bc_dual_single_gpu_matched_width_formal480_20260826_v3

echo IDENTITY
hostname
pwd
id -u
date -Is

echo FILE_SIZES
for path in \
  "$control_run/metrics.log" \
  "$method_run/metrics.log" \
  "$pair_rt/paired_resources.csv"; do
  stat -c '%s %n' "$path"
done

echo RUN_SIZES
du -sh "$control_run" "$method_run" "$control_rt" "$method_rt" "$pair_rt"

echo CHECKPOINTS
for run in "$control_run" "$method_run"; do
  echo "$run"
  find "$run" -type d -path '*/checkpoints/global_step_*' -prune -printf '%f\n' | sort -V | tail -n 8
done

echo ARTIFACTS
for run in "$control_run" "$method_run"; do
  echo "$run"
  find "$run" -maxdepth 5 -type f \( -name 'metrics.log' -o -name 'events.out.tfevents*' -o -name 'resolved_config.yaml' -o -name 'config.yaml' \) -printf '%s %p\n' | sort -k2
done

echo PROCESS
for rt in "$control_rt" "$method_rt"; do
  wrapper=$(cat "$rt/wrapper.pid")
  if test -f "$rt/driver.pid"; then
    driver=$(cat "$rt/driver.pid")
    ps -o pid=,ppid=,pgid=,stat=,etimes=,cmd= -p "$wrapper","$driver" || true
  else
    ps -o pid=,ppid=,pgid=,stat=,etimes=,cmd= -p "$wrapper" || true
    pgrep -af "ray job submit.*$(basename "$(dirname "$rt")")|rlt_single_gpu_.*matched_width_formal480_20260826_v3" | head -n 12 || true
  fi
done

echo ERRORS
for rt in "$control_rt" "$method_rt"; do
  printf '%s ' "$rt"
  for pattern in 'CUDA out of memory' 'OutOfMemoryError' 'WorkerCrashedError' 'ActorDiedError' 'NCCL error'; do
    printf '%s=%s ' "$pattern" "$(grep -cF "$pattern" "$rt/foreground.log" 2>/dev/null || true)"
  done
  echo
done

echo MEMORY
printf 'current='; cat /sys/fs/cgroup/memory.current
printf 'high='; cat /sys/fs/cgroup/memory.high
printf 'max='; cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/memory.events
grep -E 'MemTotal|MemAvailable|SwapTotal|SwapFree' /proc/meminfo

echo GPU
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
