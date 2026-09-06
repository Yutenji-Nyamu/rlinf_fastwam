set -euo pipefail
RT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-vector-render-lifecycle-fix-0008ae6
SHARED=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
test ! -e "$RT/assets"
test ! -L "$RT/assets"
test -d "$SHARED/assets"
ln -s "$SHARED/assets" "$RT/assets"
readlink -f "$RT/assets"
git -C "$RT" status --short
