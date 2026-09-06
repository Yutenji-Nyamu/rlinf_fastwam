#!/usr/bin/env bash
set -euo pipefail

control=/root/autodl-tmp/experiment_exports/rlt_single_gpu_control_success_bc_pair_smoke_20260825_v6/runtime
pair=/root/autodl-tmp/experiment_exports/rlt_success_bc_dual_single_gpu_smoke_20260825_pair_v6
control_pgid=$(cat "$control/process_group.txt")
head_pgid=$(cat "$pair/ray_head.pid")

kill -TERM -- "-$control_pgid" 2>/dev/null || true
kill -TERM -- "-$head_pgid" 2>/dev/null || true
sleep 10
kill -KILL -- "-$control_pgid" 2>/dev/null || true
kill -KILL -- "-$head_pgid" 2>/dev/null || true
sleep 2

echo "[REMAINING]"
ps -eo pid,pgid,args | grep -E '500868|498151|train_embodied_agent|raylet|gcs_server' | grep -v grep || true
echo "[GPU]"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
echo "[MEMORY_EVENTS]"
cat /sys/fs/cgroup/memory.events
