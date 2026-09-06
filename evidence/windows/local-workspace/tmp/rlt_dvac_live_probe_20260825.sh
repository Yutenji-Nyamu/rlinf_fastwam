#!/usr/bin/env bash
set -u

run_dir=/root/autodl-tmp/experiments/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1
runtime_dir=/root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1/runtime
log="$runtime_dir/foreground.log"
metrics="$run_dir/metrics.log"

echo '=== identity ==='
hostname
id -u
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
ps -eo pid,ppid,pgid,etimes,rss,stat,args --sort=pid | grep -F 'rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1' | grep -v grep || true

echo '=== core workers ==='
ps -eo pid,ppid,pgid,etimes,rss,stat,args --sort=pid | grep -E 'RLTACFSDPPolicy|MultiStepRolloutWorker|EnvWorker' | grep -v grep || true

echo '=== latest metric blocks ==='
tail -180 "$metrics" 2>/dev/null | grep -E 'Global Step:|Elapsed:|success_once=|success_at_end=|global_min_replay_size=|actor_updates_run=|critic_updates_run=|update_step=|ready_for_online=|baseline_count=|baseline_frozen=|baseline_mean=|baseline_std=|weight_|grad_norm|actor_loss|critic_loss|qf|bc_loss|alpha' | tail -100 || true

echo '=== current rollout marker ==='
tail -60 "$log" 2>/dev/null | grep -E 'Generating Rollout Epochs|Evaluating Rollout Epochs|Global Step:|Saving checkpoint' | tail -8 || true

echo '=== checkpoints ==='
find "$run_dir" -maxdepth 4 -type d -name 'global_step_*' -printf '%f %TY-%Tm-%TdT%TH:%TM:%TS\n' 2>/dev/null | sort -V | tail -25

echo '=== telemetry artifacts ==='
find "$run_dir" -type f \( -name '*dvac*' -o -name '*.npz' \) -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -25

echo '=== exact fatal counts ==='
for pattern in 'CUDA out of memory' 'NCCL error' 'NCCL timeout' 'RayTaskError' 'WorkerCrashedError' 'OutOfMemoryError'; do
  count=$(grep -cF "$pattern" "$log" 2>/dev/null || true)
  echo "$pattern=$count"
done

echo '=== last resource row ==='
head -1 "$runtime_dir/resources.csv" 2>/dev/null || true
tail -1 "$runtime_dir/resources.csv" 2>/dev/null || true

echo '=== cgroup events ==='
cat /sys/fs/cgroup/memory.current 2>/dev/null || true
cat /sys/fs/cgroup/memory.peak 2>/dev/null || true
cat /sys/fs/cgroup/memory.events 2>/dev/null || true

echo '=== gpu ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits

echo '=== disk ==='
df -h /root/autodl-tmp | tail -1
