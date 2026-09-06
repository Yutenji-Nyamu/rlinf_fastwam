set -eu
SRC=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
DST=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-vector-render-lifecycle-fix-0008ae6
BRANCH=codex/sz-robotwin-vector-render-lifecycle-fix

test "$(git -C "$SRC" rev-parse HEAD)" = "0008ae6800df9f75fc8de7098bacb01735fd8fd2"
test -z "$(git -C "$SRC" status --porcelain)"

if test -e "$DST"; then
  echo "target already exists: $DST" >&2
  exit 3
fi
if git -C "$SRC" show-ref --verify --quiet "refs/heads/$BRANCH"; then
  echo "branch already exists: $BRANCH" >&2
  exit 4
fi

git -C "$SRC" worktree add -b "$BRANCH" "$DST" 0008ae6800df9f75fc8de7098bacb01735fd8fd2
printf 'worktree=%s\n' "$DST"
git -C "$DST" status --short --branch
sha256sum "$DST/robotwin/envs/vector_env.py"
base64 -w 0 "$DST/robotwin/envs/vector_env.py"
