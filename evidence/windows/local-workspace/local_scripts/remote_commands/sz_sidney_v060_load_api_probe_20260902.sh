set -euo pipefail
src=/data/chenyiteng/projects/lerobot-sidney/lerobot-v060-py310-compat
grep -R -n -E '^def make.*pre.*post|class PI05Policy|def from_pretrained' "$src/src/lerobot/policies/pi05" "$src/src/lerobot/policies/pretrained.py" | head -40
grep -n -A35 -B5 -E '^def make.*pre.*post' "$src/src/lerobot/policies/pi05/processor_pi05.py" || true
grep -n -A90 -B5 -E 'def from_pretrained' "$src/src/lerobot/policies/pretrained.py" || true
