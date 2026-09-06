set -euo pipefail
v4=/data/chenyiteng/projects/lerobot-sidney/lerobot-0b067df57d21
sed -n '1,260p' "$v4/src/lerobot/policies/__init__.py"
echo '=== pi05 imports ==='
head -100 "$v4/src/lerobot/policies/pi05/modeling_pi05.py"
head -100 "$v4/src/lerobot/policies/pi05/processor_pi05.py"
