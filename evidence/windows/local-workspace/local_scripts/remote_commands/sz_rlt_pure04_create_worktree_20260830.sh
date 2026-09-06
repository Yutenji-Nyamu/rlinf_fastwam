set -euo pipefail

CANON=/data/chenyiteng/projects/rlinf-shenzhen/RLinf
SOURCE=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421
TARGET=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-dvac-pure-single-gpu-7d07a421
BRANCH=codex/sz-rlt-dvac-pure-single-gpu
BASE=8bbd0216113ec6eca8bbc67e79d9d642d48b8ed1
PURE=f0aaf4b71669fad38d11ac85c90670386242c29d

test "$(git -C "$SOURCE" rev-parse HEAD)" = "$BASE"
test -z "$(git -C "$SOURCE" status --porcelain)"
test ! -e "$TARGET"
if git -C "$CANON" show-ref --verify --quiet "refs/heads/$BRANCH"; then
  echo "refusing: branch already exists" >&2
  exit 20
fi

git -C "$CANON" fetch personal \
  refs/heads/codex/rlt-dvac-pure-reference-bc:refs/remotes/personal/codex/rlt-dvac-pure-reference-bc
test "$(git -C "$CANON" rev-parse personal/codex/rlt-dvac-pure-reference-bc)" = "$PURE"

git -C "$CANON" worktree add -b "$BRANCH" "$TARGET" "$BASE"

printf 'target_head='
git -C "$TARGET" rev-parse HEAD
printf 'target_branch='
git -C "$TARGET" branch --show-current
printf 'pure_ref='
git -C "$CANON" rev-parse personal/codex/rlt-dvac-pure-reference-bc
git -C "$TARGET" status --short --branch
