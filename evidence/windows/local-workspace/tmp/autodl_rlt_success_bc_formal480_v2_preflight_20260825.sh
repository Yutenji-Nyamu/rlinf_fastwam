set -euo pipefail

echo TIME
date -Is
echo IDENTITY
hostname
pwd
id -u
echo SOURCE
git -C /root/autodl-tmp/RLinf_rlt_dvac_success_bc rev-parse HEAD
git -C /root/autodl-tmp/RLinf_rlt_dvac_success_bc status --short
echo SCRIPT_SYNTAX
chmod +x \
  /tmp/autodl_run_one_rlt_success_bc_formal480_v2_20260825.sh \
  /tmp/autodl_start_shared_ray_formal480_v2_20260825.sh \
  /tmp/autodl_launch_rlt_success_bc_dual_single_gpu_formal480_v2_20260825.sh
bash -n /tmp/autodl_run_one_rlt_success_bc_formal480_v2_20260825.sh
bash -n /tmp/autodl_start_shared_ray_formal480_v2_20260825.sh
bash -n /tmp/autodl_launch_rlt_success_bc_dual_single_gpu_formal480_v2_20260825.sh
echo OK
echo LIVE_PROCESSES
pgrep -af 'train_embodied_agent|ray::|gcs_server|raylet|rlt_success_bc_dual_single_gpu_formal480' || true
echo GPU
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
echo MEMORY
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
grep MemAvailable /proc/meminfo
echo TARGETS
for target in \
  /root/autodl-tmp/experiments/rlt_single_gpu_control_formal480_20260825_v2 \
  /root/autodl-tmp/experiments/rlt_single_gpu_success_episode_bc_dvac_formal480_20260825_v2 \
  /root/autodl-tmp/experiment_exports/rlt_single_gpu_control_formal480_20260825_v2 \
  /root/autodl-tmp/experiment_exports/rlt_single_gpu_success_episode_bc_dvac_formal480_20260825_v2 \
  /root/autodl-tmp/experiment_exports/rlt_success_bc_dual_single_gpu_formal480_20260825_v2; do
  if test -e "$target"; then
    echo "EXISTS $target"
  else
    echo "ABSENT $target"
  fi
done
