set -u
echo '=== identity_time ==='
hostname
pwd
id -u
date '+%Y-%m-%d %H:%M:%S %Z'

run_dir=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822
runtime_dir=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822

echo '=== owned_processes ==='
ps -o pid,ppid,pgid,stat,etime,%cpu,%mem,rss,args -p 70610,70614,70615 2>&1 || true
pgrep -af 'idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822' || true

echo '=== progress ==='
grep -aE 'Global Step|Rollout Epoch|rollout success|Train/|Saving|checkpoint' "$runtime_dir/driver.log" 2>/dev/null | tail -n 80 || true
echo '=== log_tail ==='
tail -n 80 "$runtime_dir/driver.log" 2>/dev/null || true

echo '=== artifacts ==='
find "$run_dir" -maxdepth 3 -type d -name 'global_step_*' -printf '%f\n' 2>/dev/null | sort -V | tail -n 10
find "$run_dir" -type f -name '*.npz' 2>/dev/null | wc -l
du -sh "$run_dir" "$runtime_dir" 2>/dev/null || true

echo '=== resources ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
cat /sys/fs/cgroup/memory.current 2>/dev/null || true
cat /sys/fs/cgroup/memory.peak 2>/dev/null || true
cat /sys/fs/cgroup/memory.events 2>/dev/null || true
df -h /root/autodl-tmp | tail -n 1

echo '=== git ==='
git -C /root/autodl-tmp/RLinf_idea2_dvac_residual_downweight rev-parse HEAD
git -C /root/autodl-tmp/RLinf_idea2_dvac_residual_downweight status --short
