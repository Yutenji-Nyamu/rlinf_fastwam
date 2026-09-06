set -eu
ROOT=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
grep -RIn '^ *def close(' "$ROOT/envs" --include='*.py' 2>/dev/null | head -n 80
grep -n '^ *def close(' "$ROOT/envs/_base_task.py" 2>/dev/null || true
grep -RIn 'clear_cache_freq' "$ROOT" --include='*.py' --include='*.yml' --include='*.yaml' 2>/dev/null | head -n 100
