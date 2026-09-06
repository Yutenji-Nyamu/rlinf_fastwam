set -euo pipefail
control=/root/autodl-tmp/experiment_exports/rlt_single_gpu_control_success_bc_pair_smoke_20260825_v3/runtime
method=/root/autodl-tmp/experiment_exports/rlt_single_gpu_success_episode_bc_dvac_smoke_20260825_v3/runtime

for runtime in "$control" "$method"; do
  pgid=$(cat "$runtime/process_group.txt")
  kill -TERM -- "-$pgid" 2>/dev/null || true
done
sleep 10
for runtime in "$control" "$method"; do
  head=$(cat "$runtime/ray_head.pid")
  kill -TERM -- "-$head" 2>/dev/null || true
done
sleep 3
echo '[REMAINING]'
ps -eo pid,pgid,cmd | grep -E '423752|423753|420388|420389|train_embodied_agent|raylet|gcs_server' | grep -v grep | head -40 || true
echo '[GPU]'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
echo '[MEMORY_EVENTS]'
cat /sys/fs/cgroup/memory.events
