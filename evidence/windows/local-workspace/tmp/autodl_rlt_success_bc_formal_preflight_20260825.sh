set -u
echo '== identity =='
hostname
pwd
id -u
date -Is

echo '== gpu =='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true

echo '== relevant processes =='
ps -eo pid,pgid,etimes,cmd --sort=pid | grep -E 'raylet|gcs_server|train_embodied_agent|rlt_single_gpu|RLinf_rlt_dvac_success_bc' | grep -v grep || true

echo '== repo =='
git -C /root/autodl-tmp/RLinf_rlt_dvac_success_bc rev-parse HEAD
git -C /root/autodl-tmp/RLinf_rlt_dvac_success_bc status --short
git -C /root/autodl-tmp/RLinf_rlt_dvac_success_bc branch --show-current

echo '== formal configs =='
for path in \
  /root/autodl-tmp/RLinf_rlt_dvac_success_bc/examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_fresh480_control.yaml \
  /root/autodl-tmp/RLinf_rlt_dvac_success_bc/examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_success_episode_bc_dvac_w0to2_gpu1_fresh480.yaml
do
  test -f "$path" && echo "present $path" || echo "missing $path"
done

echo '== target dirs =='
for path in \
  /root/autodl-tmp/experiments/rlt_single_gpu_control_formal480_20260825_v1 \
  /root/autodl-tmp/experiments/rlt_single_gpu_success_episode_bc_dvac_formal480_20260825_v1 \
  /root/autodl-tmp/experiment_exports/rlt_success_bc_dual_single_gpu_formal480_20260825_v1
do
  if test -e "$path"; then echo "exists $path"; else echo "absent $path"; fi
done

echo '== memory and disk =='
awk '/MemTotal|MemAvailable/ {print}' /proc/meminfo
for path in /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.high /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.events; do
  echo "-- $path"
  cat "$path" 2>/dev/null || true
done
df -h /root/autodl-tmp
