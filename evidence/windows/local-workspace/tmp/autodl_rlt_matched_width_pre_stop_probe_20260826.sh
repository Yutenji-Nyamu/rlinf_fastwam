#!/usr/bin/env bash
set -euo pipefail

control_rt=/root/autodl-tmp/experiment_exports/rlt_single_gpu_control_formal480_20260825_v2/runtime
method_rt=/root/autodl-tmp/experiment_exports/rlt_single_gpu_success_episode_bc_dvac_formal480_20260825_v2/runtime
pair_rt=/root/autodl-tmp/experiment_exports/rlt_success_bc_dual_single_gpu_formal480_20260825_v2

echo TIME
date -Is
echo IDENTITY
hostname
pwd
id -u
echo WRAPPERS
for rt in "$control_rt" "$method_rt"; do
  pid=$(cat "$rt/wrapper.pid")
  pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
  printf '%s pid=%s pgid=%s alive=' "$rt" "$pid" "$pgid"
  kill -0 "$pid" 2>/dev/null && echo yes || echo no
  ps -o pid,ppid,pgid,stat,etime,cmd --forest -g "$pgid" || true
done
echo PAIR_PROCESSES
for name in ray_head monitor cleanup; do
  test -f "$pair_rt/$name.pid" || continue
  pid=$(cat "$pair_rt/$name.pid")
  printf '%s pid=%s alive=' "$name" "$pid"
  kill -0 "$pid" 2>/dev/null && echo yes || echo no
done
echo LATEST_STEPS
for rt in "$control_rt" "$method_rt"; do
  printf '%s ' "$rt"
  grep -oE 'Global Step: [0-9]+' "$rt/foreground.log" | tail -n 1 || true
done
echo GPU
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
echo MEMORY
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
echo SOURCE
git -C /root/autodl-tmp/RLinf_rlt_dvac_success_bc rev-parse HEAD
git -C /root/autodl-tmp/RLinf_rlt_dvac_success_bc status --short
echo RAY
/root/autodl-tmp/RLinf/.venv/bin/ray status --address="$(cat "$pair_rt/ray_address.txt")" | sed -n '1,80p'
