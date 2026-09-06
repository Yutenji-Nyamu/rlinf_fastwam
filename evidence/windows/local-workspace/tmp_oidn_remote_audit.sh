set -u
echo '=== identity/time ==='
date -Is
hostname
id

echo '=== gpu ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader

echo '=== owned training processes ==='
ps -u chenyiteng -o pid=,ppid=,lstart=,cmd= | grep -E 'ray::|fastwam|sidney|run_embodiment|launch|verl|rlinf' | grep -v grep | tail -n 120 || true

FW_WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
FW_RUN=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-formal100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-v1

echo '=== fastwam worktree ==='
git -C "$FW_WT" status --short --branch
git -C "$FW_WT" log -1 --format='%H %s'

echo '=== run source/launch files ==='
find "$FW_RUN" -maxdepth 2 -type f \( -name '*command*' -o -name '*resolved*' -o -name '*source*' -o -name '*manifest*' \) -printf '%p\n' | sort

echo '=== adapter source ==='
sed -n '1,470p' "$FW_WT/rlinf/envs/robotwin/robotwin_env.py" | tail -n 100

echo '=== likely vector env sources ==='
find /data/chenyiteng/projects/rlinf-shenzhen /data/chenyiteng/projects/robotwin-native -path '*/robotwin/envs/vector_env.py' -type f -print 2>/dev/null | sort

echo '=== likely base task sources ==='
find /data/chenyiteng/projects/rlinf-shenzhen /data/chenyiteng/projects/robotwin-native -path '*/envs/_base_task.py' -type f -print 2>/dev/null | sort
