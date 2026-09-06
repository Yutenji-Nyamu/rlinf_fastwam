#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/prism-dvac-rank-rloo
ROOT=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs
PRISM=$ROOT/prism-dvac-rank-rloo-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v1
CONTROL=$ROOT/grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2

TZ=Asia/Shanghai date '+snapshot=%F %T %Z'
printf 'worktree_head='; git -C "$WT" rev-parse HEAD
printf 'worktree_branch='; git -C "$WT" branch --show-current
echo 'worktree_status:'
git -C "$WT" status --short
echo 'remotes:'
git -C "$WT" remote -v | sed -n '1,8p'

for item in "prism:$PRISM" "control:$CONTROL"; do
  label=${item%%:*}; run=${item#*:}
  pid=$(cat "$run/runtime/wrapper.pid")
  printf '%s_wrapper pid=%s alive=' "$label" "$pid"
  if kill -0 "$pid" 2>/dev/null; then echo yes; else echo no; fi
  ps -o user=,pid=,pgid=,etimes=,stat=,args= -p "$pid" || true
  printf '%s_last=' "$label"
  grep -aE 'Global Step:|Generating Rollout Epochs:|Evaluating Rollout Epochs:|Saving checkpoint' "$run/runtime/driver.log" | tail -n1 || true
done

echo 'gpus:'
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo 'gpu_process_ownership:'
for pid in $(nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu); do
  [[ -r "/proc/$pid/environ" ]] || continue
  job=$(tr '\0' '\n' < "/proc/$pid/environ" | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
  rank=$(tr '\0' '\n' < "/proc/$pid/environ" | sed -n 's/^RANK=//p' | head -n1)
  world=$(tr '\0' '\n' < "/proc/$pid/environ" | sed -n 's/^WORLD_SIZE=//p' | head -n1)
  master=$(tr '\0' '\n' < "/proc/$pid/environ" | sed -n 's/^MASTER_PORT=//p' | head -n1)
  printf 'pid=%s user=%s job=%s rank=%s world=%s master_port=%s args=' "$pid" "$(ps -o user= -p "$pid" | xargs)" "$job" "$rank" "$world" "$master"
  ps -o args= -p "$pid" | cut -c1-160
done

echo 'prism_checkpoint:'
find "$PRISM" -type d -name global_step_10 -print -exec du -sh {} \; 2>/dev/null || true
find "$PRISM" -path '*/global_step_10/*' -type f -printf '%s %p\n' 2>/dev/null | sort -n | tail -n10 || true
echo 'memory_disk:'
awk '/MemAvailable:/ {print}' /proc/meminfo
df -h /home /data
