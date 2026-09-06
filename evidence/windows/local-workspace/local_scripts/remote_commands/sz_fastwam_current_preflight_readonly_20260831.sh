#!/usr/bin/env bash
set -u

echo '=== identity ==='
date -Is
hostname
id

echo '=== gpu summary ==='
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
echo '=== gpu compute processes ==='
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true

echo '=== host memory and pressure ==='
free -h
cat /proc/pressure/memory

echo '=== storage ==='
df -hT / /home /data

echo '=== relevant live processes (read-only) ==='
ps -eo user:16,pid,ppid,etimes,%cpu,%mem,rss,args --sort=-rss \
  | grep -E 'train_embodied_agent|ray::|raylet|gcs_server|FastWAM|RoboTwin|python.*rlinf' \
  | grep -v grep | head -n 100 || true

echo '=== current RLinf project ==='
ROOT=/data/chenyiteng/projects/rlinf-shenzhen
if test -d "$ROOT"; then
  find "$ROOT" -maxdepth 2 -mindepth 1 -type d -printf '%p\n' | sort | head -n 120
fi
for wt in \
  "$ROOT/worktrees/current-base-grpo" \
  "$ROOT/worktrees/pi05-robotwin-rl" \
  "$ROOT/worktrees/ppo-dvac-action-adv-fix" \
  "$ROOT/repo" \
  "$ROOT/RLinf"
do
  if test -e "$wt/.git"; then
    printf 'WT=%s HEAD=%s BRANCH=%s DIRTY=%s\n' \
      "$wt" \
      "$(git -C "$wt" rev-parse HEAD)" \
      "$(git -C "$wt" branch --show-current)" \
      "$(git -C "$wt" status --porcelain | wc -l)"
  fi
done

echo '=== all registered RLinf worktrees ==='
for repo in "$ROOT"/* "$ROOT"/repo "$ROOT"/RLinf; do
  if test -e "$repo/.git"; then
    git -C "$repo" worktree list --porcelain 2>/dev/null | sed -n '1,120p'
    break
  fi
done

echo '=== official Fast-WAM oracle ==='
FW=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711
if test -e "$FW/.git"; then
  printf 'FW=%s HEAD=%s BRANCH=%s DIRTY=%s\n' \
    "$FW" "$(git -C "$FW" rev-parse HEAD)" "$(git -C "$FW" branch --show-current)" \
    "$(git -C "$FW" status --porcelain | wc -l)"
fi
find /data/chenyiteng/models -maxdepth 4 -type f \
  \( -name '*.safetensors' -o -name '*.pt' -o -name '*.pth' -o -name '*.json' \) \
  -path '*FastWAM*' -printf '%s %p\n' 2>/dev/null | sort -nr | head -n 30 || true

echo '=== runtimes ==='
for py in \
  /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python \
  /home/chenyiteng/venvs/fastwam-7faa-py310-cu128/bin/python
do
  if test -x "$py"; then
    "$py" -c 'import sys,torch; print(sys.executable,sys.version.split()[0],torch.__version__,torch.version.cuda)' 2>&1
  fi
done

echo '=== shared Ray ==='
if test -x /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/ray; then
  RAY_ADDRESS=172.17.0.1:6389 /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/ray status 2>&1 | sed -n '1,100p'
fi

echo 'SZ_FASTWAM_CURRENT_PREFLIGHT_READONLY_DONE'
