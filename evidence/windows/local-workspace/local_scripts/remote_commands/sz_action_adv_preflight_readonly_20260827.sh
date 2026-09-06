#!/usr/bin/env bash
set -euo pipefail

SOURCE=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current
TARGET=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-dvac-action-adv
BRANCH=codex/sz-grpo-dvac-action-adv
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin

printf 'time='; TZ=Asia/Shanghai date --iso-8601=seconds
hostname
id
printf 'pwd='; pwd
printf 'mem_available_kib='; awk '/^MemAvailable:/ {print $2}' /proc/meminfo
df -h / /home /data
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
printf '%s\n' '--- compute processes ---'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
printf '%s\n' '--- process owners on GPU2/3 ---'
for pid in $(nvidia-smi -i 2,3 --query-compute-apps=pid --format=csv,noheader,nounits | tr -d ' ' | grep -E '^[0-9]+$' || true); do
  ps -o user=,pid=,ppid=,etimes=,cmd= -p "$pid" || true
done
printf '%s\n' '--- source ---'
git -C "$SOURCE" rev-parse HEAD
git -C "$SOURCE" branch --show-current
git -C "$SOURCE" status --short
git -C "$SOURCE" remote -v
git -C "$SOURCE" worktree list --porcelain
printf 'target_exists='; test -e "$TARGET" && echo yes || echo no
printf 'local_branch_exists='; git -C "$SOURCE" show-ref --verify --quiet "refs/heads/$BRANCH" && echo yes || echo no
printf 'remote_branch='; git -C "$SOURCE" ls-remote --heads personal "$BRANCH" || true
printf '%s\n' '--- shared ray ---'
RAY_ADDRESS=172.17.0.1:6389 "$VENV/bin/ray" status | sed -n '1,70p'
