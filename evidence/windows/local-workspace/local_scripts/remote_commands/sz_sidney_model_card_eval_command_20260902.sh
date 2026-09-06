set -euo pipefail
readme=/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab/README.md
grep -n -A40 -B8 -E 'lerobot-eval|robotwin|rename' "$readme" | head -240
