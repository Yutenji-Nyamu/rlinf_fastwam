#!/usr/bin/env bash
set -eu
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc
robotwin=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
sed -n '1,180p' "$robotwin/envs/_GLOBAL_CONFIGS.py"
sed -n '1,140p' "$root/rlinf/envs/robotwin/robotwin_env.py"
grep -rnE 'CONFIGS_PATH|config_path|os.chdir|ASSETS_PATH|load_norm_stats' "$root/rlinf/envs/robotwin" "$robotwin/robotwin/envs/vector_env.py" "$root/rlinf/models/embodiment/openpi/dataconfig"
