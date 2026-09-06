set -eu
ROOT=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
grep -RIn 'def sapien_clear_cache\|sapien_clear_cache' "$ROOT" --include='*.py' 2>/dev/null | head -n 80

echo '=== VECTOR INIT 180-360 ==='
nl -ba "$ROOT/robotwin/envs/vector_env.py" | sed -n '180,360p'
