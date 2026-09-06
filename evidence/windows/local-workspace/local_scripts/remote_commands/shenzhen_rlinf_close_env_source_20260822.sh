#!/usr/bin/env bash
set -u

ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
ROOT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-pi0-robotwin

echo '=== BASE_TASK_CLOSE_ENV ==='
nl -ba "$ROBOTWIN/envs/_base_task.py" | sed -n '620,675p'

echo '=== ROBOTWIN_ENV_INIT_CLOSE ==='
nl -ba "$ROOT/rlinf/envs/robotwin/robotwin_env.py" | sed -n '35,115p'
nl -ba "$ROOT/rlinf/envs/robotwin/robotwin_env.py" | sed -n '390,425p'

echo 'CLOSE_ENV_SOURCE_DONE'
