#!/usr/bin/env bash
set -euo pipefail

echo '=== identity ==='
date --iso-8601=seconds
hostname
pwd
id -u

echo '=== current formal run ==='
ps -p 114146,114149,114150 -o pid=,ppid=,stat=,etimes=,cmd= || true
grep -o 'Global Step:[[:space:]]*[0-9]\+/100' \
  /root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_apply_formal_100step_2gpu16env_20260821/metrics.log \
  | tail -n 3 || true

echo '=== source authority ==='
git -C /root/autodl-tmp/RLinf_idea2_dvac_train rev-parse HEAD
git -C /root/autodl-tmp/RLinf_idea2_dvac_train branch --show-current
git -C /root/autodl-tmp/RLinf_idea2_dvac_train status --short --branch
git -C /root/autodl-tmp/RLinf_idea2_dvac_train log -n 4 --oneline --decorate
git -C /root/autodl-tmp/RLinf_idea2_dvac_train remote -v

echo '=== worktrees and target ==='
git -C /root/autodl-tmp/RLinf_idea2_dvac_train worktree list --porcelain
if test -e /root/autodl-tmp/RLinf_idea2_dvac_residual_downweight; then
  echo 'TARGET_EXISTS=1'
else
  echo 'TARGET_EXISTS=0'
fi

echo '=== resources ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
df -h /root/autodl-tmp
