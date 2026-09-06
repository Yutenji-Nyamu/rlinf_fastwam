set -eu
RT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-vector-render-lifecycle-fix-0008ae6
SHARED=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
ls -ld "$RT/assets" "$SHARED/assets"
du -sh "$RT/assets" "$SHARED/assets"
printf '%s\n' 'RT top'
find "$RT/assets" -mindepth 1 -maxdepth 2 -printf '%P %y\n' | sort | head -100
printf '%s\n' 'SHARED top'
find "$SHARED/assets" -mindepth 1 -maxdepth 2 -printf '%P %y\n' | sort | head -100
git -C "$RT" status --short
