set -euo pipefail
BASE=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
TARGET=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-action-dvac-adv
BUNDLE=/data/chenyiteng/tmp/fastwam-action-dvac-adv-01d9d42a.bundle
EXPECTED_BASE=7b2331c55d14397cfb4cb16181470ddc8afae44a
test "$(git -C "$BASE" rev-parse HEAD)" = "$EXPECTED_BASE"
test -z "$(git -C "$BASE" status --porcelain)"
test -f "$BUNDLE"
git -C "$BASE" bundle verify "$BUNDLE"
printf 'base_head=%s\n' "$(git -C "$BASE" rev-parse HEAD)"
printf 'common_dir=%s\n' "$(git -C "$BASE" rev-parse --git-common-dir)"
printf 'target_exists=%s\n' "$([ -e "$TARGET" ] && echo yes || echo no)"
git -C "$BASE" remote -v
git -C "$BASE" show-ref --verify --quiet refs/heads/codex/sz-fastwam-action-dvac-adv \
  && echo branch_exists=yes || echo branch_exists=no
