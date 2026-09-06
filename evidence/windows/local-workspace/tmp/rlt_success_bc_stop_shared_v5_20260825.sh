#!/usr/bin/env bash
set -euo pipefail

control=/root/autodl-tmp/experiment_exports/rlt_single_gpu_control_success_bc_pair_smoke_20260825_v5/runtime
method=/root/autodl-tmp/experiment_exports/rlt_single_gpu_success_episode_bc_dvac_smoke_20260825_v5/runtime
pair=/root/autodl-tmp/experiment_exports/rlt_success_bc_dual_single_gpu_smoke_20260825_pair_v5

control_pgid=$(cat "$control/process_group.txt")
method_pgid=$(cat "$method/process_group.txt")
head_pgid=$(cat "$pair/ray_head.pid")
monitor_pgid=$(cat "$pair/monitor.pid")
cleanup_pgid=$(cat "$pair/cleanup.pid")

for pgid in "$control_pgid" "$method_pgid" "$head_pgid" "$monitor_pgid" "$cleanup_pgid"; do
  kill -TERM -- "-$pgid" 2>/dev/null || true
done
sleep 10
for pgid in "$control_pgid" "$method_pgid" "$head_pgid" "$monitor_pgid" "$cleanup_pgid"; do
  kill -KILL -- "-$pgid" 2>/dev/null || true
done
sleep 2

echo "[REMAINING]"
ps -eo pid,pgid,args | grep -E '474071|474072|471355|train_embodied_agent|raylet|gcs_server' | grep -v grep || true
echo "[GPU]"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
echo "[MEMORY_EVENTS]"
cat /sys/fs/cgroup/memory.events
