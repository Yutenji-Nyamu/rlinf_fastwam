set -euo pipefail
BASE=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
TARGET=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-action-dvac-adv
BUNDLE=/data/chenyiteng/tmp/fastwam-action-dvac-adv-01d9d42a.bundle
BRANCH=codex/sz-fastwam-action-dvac-adv
EXPECTED_BASE=7b2331c55d14397cfb4cb16181470ddc8afae44a
EXPECTED_HEAD=01d9d42a0d18df481b660a17aea72830ed4fb147
test "$(git -C "$BASE" rev-parse HEAD)" = "$EXPECTED_BASE"
test -z "$(git -C "$BASE" status --porcelain)"
test ! -e "$TARGET"
git -C "$BASE" fetch "$BUNDLE" "refs/heads/$BRANCH:refs/heads/$BRANCH"
test "$(git -C "$BASE" rev-parse "refs/heads/$BRANCH")" = "$EXPECTED_HEAD"
git -C "$BASE" worktree add "$TARGET" "$BRANCH"
test "$(git -C "$TARGET" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$TARGET" status --porcelain)"
git -C "$TARGET" push -u personal "$BRANCH"
printf 'installed_head=%s\n' "$(git -C "$TARGET" rev-parse HEAD)"
printf 'tracking=%s\n' "$(git -C "$TARGET" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}')"
