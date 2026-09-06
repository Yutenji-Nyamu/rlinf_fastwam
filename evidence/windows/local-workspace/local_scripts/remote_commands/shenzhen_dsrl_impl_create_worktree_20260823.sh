set -euo pipefail

BASE=/data/chenyiteng/projects/rlinf-shenzhen/RLinf
PARENT=/data/chenyiteng/projects/rlinf-current-dsrl
TARGET=/data/chenyiteng/projects/rlinf-current-dsrl/RLinf-7d07-dsrl-robotwin
BRANCH=codex/sz-current-dsrl-pi0-robotwin
PIN=7d07a4212ee6858cc333e1d4fab7a37256d1f839

date -Ins
hostname
id
test "$(git -C "$BASE" rev-parse HEAD)" = "$PIN"
test -z "$(git -C "$BASE" status --porcelain=v1 --untracked-files=all)"
test ! -e "$TARGET"
if git -C "$BASE" show-ref --verify --quiet "refs/heads/$BRANCH"; then
  echo "Refusing existing local branch: $BRANCH" >&2
  exit 1
fi
df -hT "$BASE" /data
git -C "$BASE" worktree list --porcelain

mkdir -p "$PARENT"
git -C "$BASE" worktree add -b "$BRANCH" "$TARGET" "$PIN"

test "$(git -C "$TARGET" rev-parse HEAD)" = "$PIN"
test "$(git -C "$TARGET" branch --show-current)" = "$BRANCH"
test -z "$(git -C "$TARGET" status --porcelain=v1 --untracked-files=all)"
git -C "$TARGET" status --short --branch --untracked-files=all
git -C "$TARGET" remote -v
