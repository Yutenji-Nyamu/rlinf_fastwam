set -euo pipefail
src=/data/chenyiteng/projects/lerobot-sidney/lerobot-v060-py310-compat
grep -n -A18 -B4 -E '^pi05 =|^pi =|transformers|jax|flax' "$src/pyproject.toml" || true
sed -n '1,180p' "$src/src/lerobot/policies/pi05/configuration_pi05.py"
