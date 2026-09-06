#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs
CONTROL=$ROOT/grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2
PRISM=$ROOT/prism-dvac-rank-rloo-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v1
RAYLOG=/data/chenyiteng/ray/rlt-dsrl-v3/session_latest/logs

TZ=Asia/Shanghai date '+snapshot=%F %T %Z'
echo 'control_save_timeline:'
grep -aE '\[INFO .*\] (Saving checkpoint|Saved checkpoint)|Global Step:' "$CONTROL/runtime/driver.log" | tail -n 24 || true
echo 'prism_save_timeline:'
grep -aE '\[INFO .*\] (Saving checkpoint|Saved checkpoint)|Global Step:' "$PRISM/runtime/driver.log" | tail -n 24 || true

echo 'control_step70_checkpoint:'
find "$CONTROL" -type d -name global_step_70 -print -exec du -sh {} \; 2>/dev/null || true
find "$CONTROL" -path '*/global_step_70/*' -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %f\n' 2>/dev/null | sort | tail -n 20 || true

echo 'prism_step10_checkpoint:'
find "$PRISM" -type d -name global_step_10 -print -exec du -sh {} \; 2>/dev/null || true
find "$PRISM" -path '*/global_step_10/*' -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %f\n' 2>/dev/null | sort | tail -n 20 || true

echo 'actor_fd_targets:'
for pid in 399688 399691; do
  [[ -d /proc/$pid ]] || continue
  printf 'pid=%s stdout=%s stderr=%s cwd=%s\n' "$pid" \
    "$(readlink /proc/$pid/fd/1 2>/dev/null || true)" \
    "$(readlink /proc/$pid/fd/2 2>/dev/null || true)" \
    "$(readlink /proc/$pid/cwd 2>/dev/null || true)"
done

echo 'actor_log_files:'
for pid in 399688 399691; do
  find -L "$RAYLOG" -maxdepth 1 -type f -name "*-$pid.*" -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null || true
done

echo 'actor_log_tails:'
for pid in 399688 399691; do
  for file in "$RAYLOG"/*-"$pid".*; do
    [[ -f "$file" ]] || continue
    echo "--- $file"
    tail -n 80 "$file" | grep -aEi 'checkpoint|state_dict|optim|error|warn|exception|traceback|collective|gloo|nccl|timeout' | tail -n 30 || true
  done
done

echo 'raylet_recent_relevant:'
grep -aEi '399688|399691|50000000|checkpoint|deadlock|timeout|worker.*dead|connection.*closed' "$RAYLOG/raylet.err" "$RAYLOG/raylet.out" 2>/dev/null | tail -n 60 || true
