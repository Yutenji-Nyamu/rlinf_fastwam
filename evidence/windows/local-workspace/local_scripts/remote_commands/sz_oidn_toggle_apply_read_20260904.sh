#!/usr/bin/env bash
set -eu
trial=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-oidn-toggle-20260904
rl=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
git -C "$trial" apply -
git -C "$trial" diff --check
git -C "$trial" diff -- envs/_base_task.py
sed -n '75,175p' "$trial/robotwin/envs/vector_env.py"
sed -n '465,650p' "$trial/robotwin/envs/vector_env.py"
cat "$trial/envs/_GLOBAL_CONFIGS.py"
find "$trial/envs/camera" -maxdepth 2 -type f
ls -ld /data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-vector-render-lifecycle-fix-0008ae6/assets /data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-vector-render-lifecycle-fix-0008ae6/task_config
find "$rl/toolkits" "$rl/tests" -maxdepth 5 -iname '*fastwam*'
find /data/chenyiteng/projects -maxdepth 5 -type d -iname '*fastwam*'
find /data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo -maxdepth 2 -type d
find /data/chenyiteng/projects/rlinf-shenzhen -maxdepth 3 -type f -iname '*fastwam*' | head -40
