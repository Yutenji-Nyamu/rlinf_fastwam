#!/usr/bin/env bash
set -u
old_run=/root/autodl-tmp/experiments/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1
old_runtime=/root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1/runtime
queue=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dual480_20260825_queue
control_runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_control_fresh480_20260825_v1/runtime
dvac_runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_teacher_dvac_w05to15_fresh480_20260825_v1/runtime

date -Is
echo '[QUEUE]'
if [ -f "$queue/queue.pid" ]; then ps -o pid,ppid,pgid,etimes,cmd -p "$(cat "$queue/queue.pid")"; fi
for f in old_still_running_at.txt old_completed_at.txt pair_launched_at.txt blocked.txt pair_launch_summary.txt; do
  [ -f "$queue/$f" ] && { echo "--$f"; cat "$queue/$f"; }
done
echo '[OLD]'
ps -o pid,ppid,pgid,etimes,cmd -p 106844 || true
for f in exit_code.txt finished_at.txt; do [ -f "$old_runtime/$f" ] && { echo "--$f"; cat "$old_runtime/$f"; }; done
if [ -f "$old_run/metrics.log" ]; then
  grep -oE 'Global Step:[[:space:]]+[0-9]+/480' "$old_run/metrics.log" | tail -n 1
fi
echo '[NEW]'
for runtime in "$control_runtime" "$dvac_runtime"; do
  if [ -d "$runtime" ]; then
    echo "--$runtime"
    for f in ray_cluster_resources.json wrapper.pid process_group.txt started_at.txt exit_code.txt finished_at.txt; do [ -f "$runtime/$f" ] && { echo "[$f]"; cat "$runtime/$f"; }; done
    [ -f "$runtime/foreground.log" ] && tail -n 8 "$runtime/foreground.log"
  fi
done
echo '[RESOURCES]'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
printf 'memory.current='; cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
