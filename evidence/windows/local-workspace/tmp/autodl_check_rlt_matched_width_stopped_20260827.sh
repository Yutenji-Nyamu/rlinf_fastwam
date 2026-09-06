#!/usr/bin/env bash
set -euo pipefail
control_rt=/root/autodl-tmp/experiment_exports/rlt_single_gpu_control_matched_width_formal480_20260826_v3/runtime
method_rt=/root/autodl-tmp/experiment_exports/rlt_single_gpu_success_episode_bc_dvac_matched_width_formal480_20260826_v3/runtime
pair_rt=/root/autodl-tmp/experiment_exports/rlt_success_bc_dual_single_gpu_matched_width_formal480_20260826_v3
date -Is
for rt in "$control_rt" "$method_rt"; do
  echo "RUNTIME=$rt"
  pid=$(cat "$rt/wrapper.pid")
  kill -0 "$pid" 2>/dev/null && echo "wrapper=$pid alive" || echo "wrapper=$pid stopped"
  for name in exit_code.txt finished_at.txt; do
    test -f "$rt/$name" && { printf '%s=' "$name"; cat "$rt/$name"; }
  done
done
ray_pid=$(cat "$pair_rt/ray_head.pid")
kill -0 "$ray_pid" 2>/dev/null && echo "ray_head=$ray_pid alive" || echo "ray_head=$ray_pid stopped"
echo MATCHED_PROCESSES
pgrep -af 'rlt_(control|method)_mw_f480_v3|rlt_single_gpu_(control|success_episode_bc_dvac)_matched_width_formal480_20260826_v3' || true
echo GPU
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo MEMORY
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
