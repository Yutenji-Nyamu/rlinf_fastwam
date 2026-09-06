#!/usr/bin/env bash
set -euo pipefail
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-observe-7d07a421
cd "$root"
sed -n '30,180p' rlinf/envs/robotwin/robotwin_env.py
sed -n '180,330p' rlinf/envs/robotwin/robotwin_env.py
sed -n '410,515p' rlinf/envs/robotwin/robotwin_env.py
grep -RIn 'RoboTwinEnv(' rlinf | sed -n '1,100p'
grep -RIn 'env_cls' rlinf/workers/env | sed -n '1,160p'
echo 'GPU_RAY_MODEL_SIM_USED=0'
