set -eu

ROOT=/data/chenyiteng/projects/rlinf-shenzhen
REPO=$ROOT/RLinf
TARGET=$ROOT/worktrees/sidney-pi05-current-rlinf
BASE=256eeeb4459b4bd5db85bfc6a0eb315771e8c38c
BRANCH=codex/sz-sidney-pi05-current-rlinf

test "$(git -C "$REPO" cat-file -t "$BASE")" = commit
test ! -e "$TARGET"
if git -C "$REPO" show-ref --verify --quiet "refs/heads/$BRANCH"; then
  echo "branch already exists: $BRANCH" >&2
  exit 3
fi

git -C "$REPO" worktree add -b "$BRANCH" "$TARGET" "$BASE"

test "$(git -C "$TARGET" rev-parse HEAD)" = "$BASE"
test -z "$(git -C "$TARGET" status --porcelain)"
printf 'WORKTREE=%s\nBRANCH=%s\nHEAD=%s\n' "$TARGET" "$BRANCH" "$(git -C "$TARGET" rev-parse HEAD)"
