set -euo pipefail
base=/data/chenyiteng/projects/lerobot-sidney
ls -lah "$base" 2>/dev/null || true
du -sh "$base"/* 2>/dev/null || true
find "$base" -maxdepth 2 -type d -printf '%p\n' 2>/dev/null | head -50
pgrep -af 'git clone.*lerobot-0b067|pip install.*lerobot-0b067|install-lerobot-v043' || true
