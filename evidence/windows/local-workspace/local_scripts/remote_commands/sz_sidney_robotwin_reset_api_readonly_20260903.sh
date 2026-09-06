set -euo pipefail
src=/data/chenyiteng/projects/lerobot-sidney/lerobot-v060-py310-compat/src/lerobot/envs/robotwin.py
grep -nE 'class RoboTwin|def __init__|def reset|seed|task_config|make_env|def close' "$src" | head -100
sed -n '350,510p' "$src"
sed -n '530,680p' "$src"
