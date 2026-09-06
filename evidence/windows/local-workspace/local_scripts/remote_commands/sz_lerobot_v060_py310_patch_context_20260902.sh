set -euo pipefail
s=/data/chenyiteng/projects/lerobot-sidney/lerobot-30da8e687a6d/src/lerobot
for f in datasets/streaming_dataset.py motors/motors_bus.py utils/io_utils.py configs/video.py; do
  echo "=== $f ==="
  sed -n '1,125p' "$s/$f"
done
echo '=== py310 typing additions inventory ==='
grep -R -nE 'from typing import .*(Self|override|ReadOnly|TypeIs)|from collections\.abc import .*Buffer|typing\.(Self|override|ReadOnly|TypeIs)' "$s" || true
