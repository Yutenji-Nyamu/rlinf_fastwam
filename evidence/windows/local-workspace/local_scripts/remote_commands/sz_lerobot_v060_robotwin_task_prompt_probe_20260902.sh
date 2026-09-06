set -euo pipefail
src=/data/chenyiteng/projects/lerobot-sidney/lerobot-v060-py310-compat/src/lerobot/envs/robotwin.py
grep -n -A8 -B8 -E 'task_description|instruction|"task"|task=' "$src" | head -160
