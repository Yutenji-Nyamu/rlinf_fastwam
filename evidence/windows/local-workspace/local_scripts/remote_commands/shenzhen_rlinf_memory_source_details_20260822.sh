#!/usr/bin/env bash
set -u

ROOT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-pi0-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support

echo '=== RECORDVIDEO_300_490 ==='
nl -ba "$ROOT/rlinf/envs/wrappers/record_video.py" | sed -n '300,490p'

echo '=== ENVWORKER_720_790 ==='
nl -ba "$ROOT/rlinf/workers/env/env_worker.py" | sed -n '720,790p'

echo '=== VECTOR_ENV_1_280 ==='
nl -ba "$ROBOTWIN/robotwin/envs/vector_env.py" | sed -n '1,280p'

echo '=== VECTOR_ENV_280_460 ==='
nl -ba "$ROBOTWIN/robotwin/envs/vector_env.py" | sed -n '280,460p'

echo '=== ROBOTWIN_ENV_KEY_AREAS ==='
grep -n -E 'VectorEnv|make_env|reset|close|RecordVideo|capture_image|render' "$ROOT/rlinf/envs/robotwin/robotwin_env.py" | head -n 200

echo 'MEMORY_SOURCE_DETAILS_DONE'
