set -u
SRC=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
DST=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-vector-render-lifecycle-fix-0008ae6
git -C "$SRC" rev-parse HEAD
git -C "$SRC" status --short --branch
if test -e "$DST"; then
  echo EXISTS
  git -C "$DST" status --short --branch
fi
git -C "$SRC" branch --list 'codex/sz-robotwin-vector-render-lifecycle-fix'
