set -eu
date -Is
id
hostname
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc
git -C "$root" rev-parse HEAD
git -C "$root" branch --show-current
git -C "$root" status --porcelain
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits
free -h
df -h /data /home
cat /proc/pressure/memory
ps -u chenyiteng -o pid,ppid,pgid,etime,args | grep -E 'train_embodied|wrapper.sh|gcs_server|raylet' | grep -v grep || true
sha256sum "$root/rlinf/workers/env/env_worker.py" "$root/rlinf/envs/robotwin/robotwin_env.py" "$root/examples/embodiment/config/robotwin_adjust_bottle_online_bc_openpi.yaml" "$root/tests/unit_tests/test_online_bc.py"
