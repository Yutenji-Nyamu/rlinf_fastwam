set -euo pipefail
printf 'IDENTITY\n'
hostname
pwd
id -u
date --iso-8601=seconds
printf 'CONTAINER_START\n'
ps -p 1 -o lstart=,cmd=
printf 'GPU\n'
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits
printf 'MATCHED_PROCESSES\n'
pgrep -af 'rlt_teacher_dvac|rlt_stage2|ray::|raylet|gcs_server|idea2_dvac' || true
printf 'CGROUP\n'
for f in memory.current memory.high memory.max memory.swap.current memory.swap.max memory.events memory.stat memory.pressure; do
  printf '%s\n' "---$f"
  cat "/sys/fs/cgroup/$f"
done
printf 'MEMORY_RECLAIM_NODE\n'
ls -l /sys/fs/cgroup/memory.reclaim || true
printf 'VM_TUNABLES\n'
for f in /proc/sys/vm/swappiness /proc/sys/vm/vfs_cache_pressure /proc/sys/vm/overcommit_memory /sys/kernel/mm/transparent_hugepage/enabled; do
  printf '%s=' "$f"
  cat "$f"
done
printf 'HOST_MEMORY\n'
free -h
printf 'DISK\n'
df -h /root/autodl-tmp
printf 'RECLAIM_REFERENCES\n'
rg -n --hidden --glob '!*.git/*' 'memory\.reclaim|drop_caches|memory\.high|memory\.max|RAY_memory_usage_threshold|RAY_memory_monitor_refresh_ms' \
  /root/autodl-tmp/RLinf_rlt_teacher_dvac \
  /root/autodl-tmp/experiment_exports 2>/dev/null | head -n 200 || true
printf 'REPO\n'
cd /root/autodl-tmp/RLinf_rlt_teacher_dvac
git branch --show-current
git rev-parse HEAD
git status --short
git rev-list --left-right --count '@{upstream}...HEAD'
printf 'DVAC_CONFIG\n'
sed -n '1,260p' examples/embodiment/config/robotwin/adjust_bottle_rlt_stage2_8env250_teacher_dvac_w0to2.yaml
printf 'BASE_CONFIG\n'
sed -n '1,320p' examples/embodiment/config/robotwin/adjust_bottle_rlt_stage2_8env250.yaml
printf 'HISTORICAL_RESUME_CONFIG\n'
find /root/autodl-tmp/RLinf_rlt_pi0_robotwin -type f -name '*resume*480*.yaml' -o -name '*8env*250*.yaml' | sort
