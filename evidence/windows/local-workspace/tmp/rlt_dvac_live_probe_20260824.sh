#!/usr/bin/env bash
set -u

run_dir=/root/autodl-tmp/experiments/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1
runtime_dir=/root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1/runtime

echo '=== identity ==='
hostname
pwd
id -u
date -Is

echo '=== lifecycle ==='
for name in started_at finished_at exit_code driver.pid wrapper.pid monitor.pid; do
  path="$runtime_dir/$name"
  if [ -f "$path" ]; then
    printf '%s=' "$name"
    tr '\n' ' ' < "$path"
    echo
  else
    echo "$name=missing"
  fi
done

echo '=== exact run processes ==='
ps -eo pid,ppid,pgid,etimes,rss,stat,args --sort=pid | grep -F 'rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1' | grep -v grep || true

echo '=== worker summary ==='
ps -eo pid,ppid,pgid,etimes,rss,stat,args --sort=pid | grep -E 'RLTACFSDPPolicy|MultiStepRolloutWorker|EnvWorker' | grep -v grep || true

echo '=== runtime files ==='
find "$runtime_dir" -maxdepth 3 -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -40

echo '=== run top-level ==='
find "$run_dir" -maxdepth 3 -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -60

echo '=== foreground progress ==='
log="$runtime_dir/foreground.log"
if [ -f "$log" ]; then
  grep -E 'Global Step|global_step|success_once|ready_for_online|actor_update|critic_update|eval|Eval|checkpoint|Saving|saved|DVAC|dvac|Traceback|CUDA out of memory|NCCL|RayTaskError|WorkerCrashed|memory pressure' "$log" | tail -160
  echo '=== foreground tail ==='
  tail -80 "$log"
else
  echo 'foreground.log missing'
fi

echo '=== checkpoints ==='
find "$run_dir" -maxdepth 4 -type d -name 'global_step_*' -printf '%f %TY-%Tm-%TdT%TH:%TM:%TS\n' 2>/dev/null | sort -V | tail -30

echo '=== resources latest ==='
resource_csv="$runtime_dir/resource_monitor/resources.csv"
if [ -f "$resource_csv" ]; then
  head -2 "$resource_csv"
  tail -20 "$resource_csv"
else
  echo 'resources.csv missing'
fi

echo '=== memory events ==='
cat /sys/fs/cgroup/memory.current 2>/dev/null || true
cat /sys/fs/cgroup/memory.peak 2>/dev/null || true
cat /sys/fs/cgroup/memory.events 2>/dev/null || true

echo '=== gpu ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits

echo '=== disk ==='
df -h /root/autodl-tmp
