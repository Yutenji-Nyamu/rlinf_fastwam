set -euo pipefail
echo '=== identity ==='
hostname
pwd
id -u
date -Is
echo '=== gpu ==='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo '=== memory ==='
free -h
if [ -r /sys/fs/cgroup/memory.current ]; then
  printf 'cgroup_current='; cat /sys/fs/cgroup/memory.current
  printf 'cgroup_max='; cat /sys/fs/cgroup/memory.max
  cat /sys/fs/cgroup/memory.events
fi
echo '=== disk ==='
df -h /root/autodl-tmp
echo '=== relevant processes ==='
ps -eo pid,ppid,pgid,etimes,cmd --sort=pid | grep -E 'train_embodied_agent|raylet|gcs_server|RLinf|rlt' | grep -v grep || true
echo '=== worktree ==='
repo=/root/autodl-tmp/RLinf_rlt_teacher_dvac
test -e "$repo/.git"
git -C "$repo" status --short
git -C "$repo" branch --show-current
git -C "$repo" rev-parse HEAD
git -C "$repo" remote -v
git -C "$repo" log -5 --oneline
echo '=== candidate configs/runtime ==='
find "$repo/examples/embodiment/config" -maxdepth 1 -type f -name '*single_gpu*' -printf '%f\n' | sort
find /root/autodl-tmp -maxdepth 2 -type d -name '*single_gpu*' -printf '%p\n' | sort | head -80
echo '=== ray temp dirs ==='
find /root/autodl-tmp -maxdepth 2 -type d -name 'ray*' -printf '%p\n' | sort | head -80
