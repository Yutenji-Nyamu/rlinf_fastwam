set -u

WT=/data/chenyiteng/projects/rlinf-current-dsrl/RLinf-7d07-dsrl-robotwin
CFG="$WT/examples/embodiment/config/robotwin_adjust_bottle_dsrl_openpi.yaml"
ENTRY="$WT/examples/embodiment/train_embodied_agent.py"
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
RUN_ROOT=/data/chenyiteng/results/rlinf-current-dsrl/smoke-dsrl-pi0-robotwin-2gpu4env-warm4-20260823

echo '=== identity/time ==='
date -Is
id
hostname

echo '=== gpu summary ==='
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,pstate --format=csv,noheader
echo '=== gpu compute apps ==='
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader 2>&1 || true

echo '=== related processes ==='
ps -eo user,pid,ppid,lstart,etime,rss,stat,args --sort=-rss | awk 'NR==1 || /train_embodied_agent|raylet|gcs_server|dashboard_agent|runtime_env_agent|robotwin_adjust_bottle_(dsrl|grpo)|rlinf-current-(dsrl|rlt)|monitor_resources/ {print}' | head -n 100

echo '=== host memory ==='
free -h
grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree|CommitLimit|Committed_AS):' /proc/meminfo
cat /proc/pressure/memory
vmstat 1 3

echo '=== storage ==='
df -hT / /home /data
df -i / /home /data

echo '=== worktree live ==='
git -C "$WT" rev-parse HEAD
git -C "$WT" branch --show-current
git -C "$WT" status --short --branch
git -C "$WT" rev-list --left-right --count HEAD...@{upstream} 2>&1 || true
git -C "$WT" log -1 --format='%H%n%aI%n%s'

echo '=== immutable input probes ==='
stat -c '%n|%s|%y' "$CFG" "$ENTRY" "$PY"
sha256sum "$CFG" "$ENTRY"
test -d "$MODEL" && echo "MODEL_OK=$MODEL" || echo "MODEL_MISSING=$MODEL"
test -d "$ROBOTWIN" && echo "ROBOTWIN_OK=$ROBOTWIN" || echo "ROBOTWIN_MISSING=$ROBOTWIN"
du -sh "$MODEL" "$ROBOTWIN" 2>&1

echo '=== proposed outputs ==='
if test -e "$RUN_ROOT"; then
  echo "RUN_ROOT_EXISTS=$RUN_ROOT"
  find "$RUN_ROOT" -maxdepth 2 -printf '%y %p %s\n' | head -n 50
else
  echo "RUN_ROOT_ABSENT=$RUN_ROOT"
fi

echo '=== live config ==='
sed -n '1,260p' "$CFG"

