set -eu
ROOT=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
VENV=/data/chenyiteng/envs/rlinf-fastwam-current
grep -n '^import sapien\|^from sapien' "$ROOT/envs/_base_task.py" | head -20
grep -n 'def close_env' "$ROOT/envs/_base_task.py"
sed -n '630,675p' "$ROOT/envs/_base_task.py"
"$VENV/bin/python" - <<'PY'
import sapien
print(sapien.__version__)
print(sapien.render.clear_cache)
PY
