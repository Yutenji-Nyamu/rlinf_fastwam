set -euo pipefail

TARGET=/data/chenyiteng/models/fasterwam/release-6bf9471
find "$TARGET" -type f -printf '%s %p\n' | sort -n
pgrep -af 'wget|step_029355|snapshot_download' || true
