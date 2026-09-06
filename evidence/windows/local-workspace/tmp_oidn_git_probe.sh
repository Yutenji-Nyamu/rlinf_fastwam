set -eu
RT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-vector-render-lifecycle-fix-0008ae6
git -C "$RT" remote -v
git -C "$RT" config --get user.name || true
git -C "$RT" config --get user.email || true
git -C "$RT" status --short --branch
git -C "$RT" diff --check
