set -euo pipefail
control=/root/autodl-tmp/experiment_exports/rlt_single_gpu_control_success_bc_pair_smoke_20260825_v4/runtime
method=/root/autodl-tmp/experiment_exports/rlt_single_gpu_success_episode_bc_dvac_smoke_20260825_v4/runtime
pair=/root/autodl-tmp/experiment_exports/rlt_success_bc_dual_single_gpu_smoke_20260825_pair_v4
for runtime in "$control" "$method"; do
  kill -TERM -- "-$(cat "$runtime/process_group.txt")" 2>/dev/null || true
done
kill -TERM -- "-$(cat "$pair/ray_head.pid")" 2>/dev/null || true
kill -TERM -- "-$(cat "$pair/monitor.pid")" 2>/dev/null || true
kill -TERM -- "-$(cat "$pair/cleanup.pid")" 2>/dev/null || true
sleep 10
for pgid in \
  "$(cat "$control/process_group.txt")" \
  "$(cat "$method/process_group.txt")" \
  "$(cat "$pair/ray_head.pid")" \
  "$(cat "$pair/monitor.pid")" \
  "$(cat "$pair/cleanup.pid")"; do
  kill -KILL -- "-$pgid" 2>/dev/null || true
done
sleep 2
ps -eo pid,pgid,cmd | grep -E '445269|445270|442553|train_embodied_agent|raylet|gcs_server' | grep -v grep | head -30 || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
cat /sys/fs/cgroup/memory.events
