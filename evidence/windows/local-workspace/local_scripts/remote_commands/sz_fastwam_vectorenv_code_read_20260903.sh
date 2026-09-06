set -eu
ROOT=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support

echo '=== VECTOR RESET/CLOSE 1-180 ==='
nl -ba "$ROOT/robotwin/envs/vector_env.py" | sed -n '1,180p'
echo '=== VECTOR RESET 360-430 ==='
nl -ba "$ROOT/robotwin/envs/vector_env.py" | sed -n '360,430p'
echo '=== BASE CLOSE ==='
nl -ba "$ROOT/envs/_base_task.py" | sed -n '610,680p'
echo '=== ROBOTWIN ADAPTER OFFLOAD ==='
nl -ba /data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo/rlinf/envs/robotwin/robotwin_env.py 2>/dev/null | sed -n '390,425p' || true
