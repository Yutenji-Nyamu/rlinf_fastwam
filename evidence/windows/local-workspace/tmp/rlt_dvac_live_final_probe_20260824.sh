#!/usr/bin/env bash
set -u

run_dir=/root/autodl-tmp/experiments/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1
runtime_dir=/root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1/runtime
log="$runtime_dir/foreground.log"

echo '=== time ==='
date -Is
echo '=== lifecycle ==='
for name in started_at.txt finished_at.txt exit_code.txt driver_pid.txt monitor_pid.txt; do
  if [ -f "$runtime_dir/$name" ]; then
    printf '%s=' "$name"
    tr '\n' ' ' < "$runtime_dir/$name"
    echo
  else
    echo "$name=missing"
  fi
done
echo '=== owned processes ==='
ps -eo pid,ppid,pgid,etimes,rss,stat,args | grep -F 'rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1' | grep -v grep || true
echo '=== latest completed metric ==='
grep 'Global Step:' "$run_dir/metrics.log" | tail -1
echo '=== current rollout marker ==='
tail -40 "$log" | grep -E 'Generating Rollout Epochs|Global Step:' | tail -4 || true
echo '=== checkpoints ==='
find "$run_dir" -maxdepth 4 -type d -name 'global_step_*' -printf '%f\n' 2>/dev/null | sort -V | tail -5
echo '=== exact fatal counts ==='
for pattern in 'CUDA out of memory' 'NCCL error' 'NCCL timeout' 'RayTaskError' 'WorkerCrashedError' 'OutOfMemoryError'; do
  count=$(grep -cF "$pattern" "$log" 2>/dev/null || true)
  echo "$pattern=$count"
done
echo '=== last resource row ==='
tail -1 "$runtime_dir/resources.csv"
echo '=== cgroup events ==='
cat /sys/fs/cgroup/memory.events 2>/dev/null || true
echo '=== gpu ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo '=== disk ==='
df -h /root/autodl-tmp | tail -1
