#!/usr/bin/env bash
set -euo pipefail

p03_rt=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dvac_pure03_reference_bc_s1p0_formal480_20260829_v1/runtime
p04_rt=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dvac_pure04_reference_bc_s1p5_formal480_20260829_v1/runtime
pair_rt=/root/autodl-tmp/experiment_exports/rlt_dvac_pure03_pure04_dual_single_gpu_formal480_20260829_v1
driver=/tmp/autodl_run_one_rlt_pure_formal480_20260828.sh

date -Is | tee "$pair_rt/user_stop_requested_at.txt"

for rt in "$p03_rt" "$p04_rt"; do
  pid=$(cat "$rt/wrapper.pid")
  pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
  test "$pid" = "$pgid"
  ps -o args= -p "$pid" | grep -F "$driver" >/dev/null
  echo "STOP_TARGET pid=$pid pgid=$pgid"
  ps -o pid=,ppid=,pgid=,stat=,etimes=,cmd= -p "$pid"
done

p03_pid=$(cat "$p03_rt/wrapper.pid")
p04_pid=$(cat "$p04_rt/wrapper.pid")
kill -INT -- -"$p03_pid" -"$p04_pid"

for _ in $(seq 1 15); do
  alive=0
  kill -0 "$p03_pid" 2>/dev/null && alive=1
  kill -0 "$p04_pid" 2>/dev/null && alive=1
  test "$alive" = 0 && break
  sleep 2
done
kill -0 "$p03_pid" 2>/dev/null && kill -TERM -- -"$p03_pid" || true
kill -0 "$p04_pid" 2>/dev/null && kill -TERM -- -"$p04_pid" || true

for name in monitor cleanup; do
  test -f "$pair_rt/$name.pid" || continue
  pid=$(cat "$pair_rt/$name.pid")
  kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" || true
done

ray_pid=$(cat "$pair_rt/ray_head.pid")
if kill -0 "$ray_pid" 2>/dev/null; then
  ray_pgid=$(ps -o pgid= -p "$ray_pid" | tr -d ' ')
  test "$ray_pid" = "$ray_pgid"
  ps -o args= -p "$ray_pid" | grep -E 'ray start.*--port=52001' >/dev/null
  kill -INT -- -"$ray_pid" || true
  for _ in $(seq 1 10); do
    kill -0 "$ray_pid" 2>/dev/null || break
    sleep 2
  done
  kill -0 "$ray_pid" 2>/dev/null && kill -TERM -- -"$ray_pid" || true
fi

sleep 5
echo AFTER
date -Is
for pid in "$p03_pid" "$p04_pid" "$ray_pid"; do
  if kill -0 "$pid" 2>/dev/null; then echo "pid=$pid alive=yes"; else echo "pid=$pid alive=no"; fi
done
for rt in "$p03_rt" "$p04_rt"; do
  echo "RUNTIME=$rt"
  for name in exit_code.txt finished_at.txt; do
    if test -f "$rt/$name"; then printf '%s=' "$name"; cat "$rt/$name"; fi
  done
done
echo TARGET_PROCESSES
ps -eo pid=,ppid=,pgid=,stat=,cmd= | grep -E 'rlt_pure03_s10_f480_v1|rlt_pure04_s15_f480_v1|ray_shared_52001|gcs_server.*52001|raylet.*52001' | grep -v grep || true
echo GPU
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo MEMORY
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
