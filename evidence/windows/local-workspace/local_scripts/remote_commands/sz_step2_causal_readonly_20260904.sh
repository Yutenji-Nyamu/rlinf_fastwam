#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
date -Is
id
base=/data/chenyiteng/projects/rlinf-shenzhen/worktrees
git -C "$base/robotwin-clean-oidn-off-20260904" status --short
git -C "$base/robotwin-clean-oidn-off-20260904" rev-parse HEAD
git -C "$base/robotwin-clean-oidn-off-20260904" diff 0008ae6800df9f75fc8de7098bacb01735fd8fd2 HEAD -- envs/_base_task.py robotwin/envs/vector_env.py
git -C "$base/robotwin-vector-render-lifecycle-fix-0008ae6" diff 0008ae6800df9f75fc8de7098bacb01735fd8fd2 HEAD -- robotwin/envs/vector_env.py
git -C "$base/fastwam-current-grpo" status --short
git -C "$base/fastwam-current-grpo" rev-parse HEAD
sed -n '260,355p' "$base/robotwin-clean-oidn-off-20260904/envs/camera/camera.py"
grep -n -A65 -B6 'def get_obs\|def _update_render\|def close' "$base/robotwin-clean-oidn-off-20260904/envs/_base_task.py"
sed -n '1,200p' "$base/robotwin-clean-oidn-off-20260904/robotwin/envs/vector_env.py"
timeout 12 /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/py-spy dump --pid 1053120 --nonblocking
date -Is
