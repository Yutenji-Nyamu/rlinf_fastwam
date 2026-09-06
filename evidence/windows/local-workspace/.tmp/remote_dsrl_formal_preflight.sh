set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
PY=/root/autodl-tmp/RLinf/.venv/bin/python
ROBOTWIN=/root/autodl-tmp/RoboTwin_RLinf
MODEL=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle
CFG="$REPO/examples/embodiment/config/robotwin_adjust_bottle_dsrl_openpi.yaml"
RUN_ROOT="$REPO/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1"

echo "TIME=$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "IDENTITY=$(id -un)@$(hostname)"
echo "BRANCH=$(git -C "$REPO" branch --show-current)"
echo "HEAD=$(git -C "$REPO" rev-parse HEAD)"
echo "UPSTREAM=$(git -C "$REPO" rev-parse '@{upstream}')"
if test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"; then
  echo "WORKTREE_CLEAN=1"
else
  echo "WORKTREE_CLEAN=0"
  git -C "$REPO" status --short
fi
echo "CONFIG_SHA256=$(sha256sum "$CFG" | awk '{print $1}')"
test -x "$PY"
test -d "$ROBOTWIN"
test -d "$MODEL"
test -s "$MODEL/physical-intelligence/robotwin/norm_stats.json"
test -s "$CFG"
if test -e "$RUN_ROOT"; then
  echo "RUN_ROOT_EXISTS=1"
else
  echo "RUN_ROOT_EXISTS=0"
fi

echo "PROCESS_SCAN"
pgrep -x raylet || true
pgrep -x gcs_server || true
pgrep -af '^/root/autodl-tmp/RLinf/.venv/bin/python .*train_embodied_agent.py|^/root/autodl-tmp/RLinf/.venv/bin/python .*monitor_resources.py|^ray::' || true

echo "GPU_STATE"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true

echo "CGROUP_STATE"
printf "memory.current="
cat /sys/fs/cgroup/memory.current
printf "memory.max="
cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/memory.events
grep -E '^(anon|file|shmem|file_mapped|inactive_file|active_file|kernel_stack|pagetables|slab) ' /sys/fs/cgroup/memory.stat
cat /sys/fs/cgroup/memory.pressure

echo "FILESYSTEM_STATE"
df -h /root/autodl-tmp /dev/shm
du -sh "$REPO/logs/20260728_dsrl_pi0_robotwin_n20_smoke_v1"
echo "FORMAL_PREFLIGHT_COMPLETE=1"
