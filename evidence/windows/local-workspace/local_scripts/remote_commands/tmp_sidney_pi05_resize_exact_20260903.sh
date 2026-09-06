set -euo pipefail
LR=/data/chenyiteng/projects/lerobot-sidney/lerobot-v060-py310-compat
cd "$LR"
sed -n '145,245p' src/lerobot/policies/pi05/modeling_pi05.py
