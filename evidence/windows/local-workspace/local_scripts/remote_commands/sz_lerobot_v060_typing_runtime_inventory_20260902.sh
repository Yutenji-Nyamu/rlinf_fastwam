set -euo pipefail
src=/data/chenyiteng/projects/lerobot-sidney/lerobot-v060-py310-compat/src/lerobot
grep -R -n -E '^from typing import .*(Unpack|Self|NotRequired|Required|TypeAliasType|override)' "$src" --include='*.py' || true
