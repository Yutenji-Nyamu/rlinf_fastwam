#!/usr/bin/env bash
set -euo pipefail

control_rt=/root/autodl-tmp/experiment_exports/rlt_single_gpu_control_formal480_20260825_v2/runtime
method_rt=/root/autodl-tmp/experiment_exports/rlt_single_gpu_success_episode_bc_dvac_formal480_20260825_v2/runtime
pair_rt=/root/autodl-tmp/experiment_exports/rlt_success_bc_dual_single_gpu_formal480_20260825_v2

echo BEFORE
date -Is
for rt in "$control_rt" "$method_rt"; do
  pid=$(cat "$rt/wrapper.pid")
  pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
  test "$pid" = "$pgid"
  ps -o args= -p "$pid" | grep -F '/tmp/autodl_run_one_rlt_success_bc_formal480_v2_20260825.sh' >/dev/null
  echo "target pid=$pid pgid=$pgid"
  tail -n 12 "$rt/foreground.log" || true
done

control_pid=$(cat "$control_rt/wrapper.pid")
method_pid=$(cat "$method_rt/wrapper.pid")
kill -INT -- -"$control_pid" -"$method_pid"

for _ in $(seq 1 20); do
  alive=0
  kill -0 "$control_pid" 2>/dev/null && alive=1
  kill -0 "$method_pid" 2>/dev/null && alive=1
  test "$alive" = 0 && break
  sleep 2
done
kill -0 "$control_pid" 2>/dev/null && kill -TERM -- -"$control_pid" || true
kill -0 "$method_pid" 2>/dev/null && kill -TERM -- -"$method_pid" || true

for name in monitor cleanup; do
  test -f "$pair_rt/$name.pid" || continue
  pid=$(cat "$pair_rt/$name.pid")
  kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" || true
done

ray_pid=$(cat "$pair_rt/ray_head.pid")
if kill -0 "$ray_pid" 2>/dev/null; then
  ray_pgid=$(ps -o pgid= -p "$ray_pid" | tr -d ' ')
  test "$ray_pid" = "$ray_pgid"
  kill -INT -- -"$ray_pid" || true
  for _ in $(seq 1 10); do
    kill -0 "$ray_pid" 2>/dev/null || break
    sleep 2
  done
  kill -0 "$ray_pid" 2>/dev/null && kill -TERM -- -"$ray_pid" || true
fi

sleep 3
echo AFTER
date -Is
for pid in "$control_pid" "$method_pid" "$ray_pid"; do
  printf 'pid=%s alive=' "$pid"
  kill -0 "$pid" 2>/dev/null && echo yes || echo no
done
echo OLD_OUTPUTS
du -sh \
  /root/autodl-tmp/experiments/rlt_single_gpu_control_formal480_20260825_v2 \
  /root/autodl-tmp/experiments/rlt_single_gpu_success_episode_bc_dvac_formal480_20260825_v2
echo CHECKPOINTS
find \
  /root/autodl-tmp/experiments/rlt_single_gpu_control_formal480_20260825_v2 \
  /root/autodl-tmp/experiments/rlt_single_gpu_success_episode_bc_dvac_formal480_20260825_v2 \
  -type d -path '*/checkpoints/global_step_*' -printf '%p\n' | sort -V | tail -n 16
echo GPU
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
echo MEMORY_EVENTS
cat /sys/fs/cgroup/memory.events
