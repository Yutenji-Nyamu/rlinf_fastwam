#!/usr/bin/env bash
set -euo pipefail
run=/root/autodl-tmp/experiments/rlt_single_gpu_dvac_pure_reference_bc_s0p5_smoke_20260827_v2
runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dvac_pure_reference_bc_s0p5_smoke_20260827_v2/runtime
launcher_pid=$(cat /tmp/rlt_dvac_pure_smoke_launcher_20260827_v2.pid)
date -Is
kill -0 "$launcher_pid" 2>/dev/null && echo "launcher=$launcher_pid alive" || echo "launcher=$launcher_pid stopped"
for name in driver.pid ray_head.pid exit_code.txt started_at.txt finished_at.txt source_head.txt; do
  if test -f "$runtime/$name"; then printf '%s=' "$name"; cat "$runtime/$name"; fi
done
if test -f "$runtime/driver.pid"; then
  driver=$(cat "$runtime/driver.pid")
  kill -0 "$driver" 2>/dev/null && echo "driver=$driver alive" || echo "driver=$driver stopped"
fi
echo PROGRESS
grep -E 'Global Step:|Start running|Rollout Epoch|rlt_dvac|success_once|Traceback|RuntimeError|CUDA out of memory' "$runtime/foreground.log" 2>/dev/null | tail -n 80 || true
echo ARTIFACTS
find "$run" -maxdepth 8 -type f \( -name 'metrics.log' -o -name 'update_*.npz' -o -name 'complete.json' \) -printf '%s %p\n' 2>/dev/null | tail -n 30 || true
echo GPU
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo MEMORY
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
