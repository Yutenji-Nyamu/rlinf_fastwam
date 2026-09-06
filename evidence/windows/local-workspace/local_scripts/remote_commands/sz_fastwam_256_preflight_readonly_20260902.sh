set -u

OLD_RUN=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-formal100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-v1
OLD_PACKET=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/packets/fastwam-grpo-control-formal100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-v1
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo

date '+TIME %Y-%m-%d %H:%M:%S %Z'
printf 'IDENTITY\n'
id
printf 'GPU\n'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null || true
printf 'MEMORY_DISK\n'
free -h
df -h / /home /data
printf 'RAY_NAMESPACES\n'
ps -eo user,pid,ppid,pgid,etimes,rss,stat,args --sort=pid | grep -E 'ray::|gcs_server|raylet|fastwam-grpo|pi05' | grep -v grep | tail -220 || true
printf 'SOURCE\n'
git -C "$WT" rev-parse HEAD
git -C "$WT" status --short --branch
printf 'OLD_PACKET_FILES\n'
find "$OLD_PACKET" -maxdepth 2 -type f -printf '%P %s\n' 2>/dev/null | sort
printf 'OLD_CONTRACT\n'
sed -n '1,240p' "$OLD_PACKET/contract.json" 2>/dev/null || true
printf 'OLD_RESOLVED_CONFIG\n'
sed -n '1,320p' "$OLD_PACKET/resolved.yaml" 2>/dev/null || true
printf 'OLD_WRAPPER\n'
sed -n '1,260p' "$OLD_RUN/runtime/wrapper.sh" 2>/dev/null || true
printf 'OLD_OBSERVER\n'
sed -n '1,240p' "$OLD_RUN/runtime/observer.sh" 2>/dev/null || true
printf 'OLD_RUNTIME_FILES\n'
find "$OLD_RUN/runtime" -maxdepth 1 -type f -printf '%f %s\n' 2>/dev/null | sort
