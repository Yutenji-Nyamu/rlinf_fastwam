set -euo pipefail
src=/data/chenyiteng/projects/lerobot-sidney/lerobot-v060-py310-compat
sed -n '270,350p' "$src/src/lerobot/policies/factory.py"
sed -n '940,1055p' "$src/src/lerobot/policies/pi05/modeling_pi05.py"
