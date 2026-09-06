set -euo pipefail
RT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-vector-render-lifecycle-fix-0008ae6
SHARED=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
for name in background_texture embodiments objects; do
  test -d "$SHARED/assets/$name"
  test ! -e "$RT/assets/$name"
done
for name in background_texture embodiments objects; do
  ln -s "$SHARED/assets/$name" "$RT/assets/$name"
done
for name in background_texture embodiments objects; do
  printf '%s -> %s\n' "$RT/assets/$name" "$(readlink -f "$RT/assets/$name")"
done
git -C "$RT" status --short
