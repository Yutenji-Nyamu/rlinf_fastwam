set -euo pipefail
hostname
pwd
id -u
echo PROCESS_MATCHES
pgrep -af 'ray|rlinf|fsdp_rlt|launch_shared_dual_smoke|run_one_shared_smoke' || true
echo GPU_STATE
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo MEMORY_EVENTS
cat /sys/fs/cgroup/memory.events
echo WORKTREE
git -C /root/autodl-tmp/RLinf_rlt_dvac_success_bc status --short
git -C /root/autodl-tmp/RLinf_rlt_dvac_success_bc rev-parse HEAD
