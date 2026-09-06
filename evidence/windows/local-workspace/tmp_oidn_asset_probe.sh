set -u
RT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-vector-render-lifecycle-fix-0008ae6
SHARED=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
for p in "$RT/assets" "$RT/robotwin/assets" "$SHARED/assets" "$SHARED/robotwin/assets"; do
  if test -e "$p" || test -L "$p"; then
    printf 'exists %s -> %s\n' "$p" "$(readlink -f "$p")"
  else
    printf 'missing %s\n' "$p"
  fi
done
find "$SHARED" -maxdepth 3 -path '*/assets/objects/objaverse/list.json' -print
