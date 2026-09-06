set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-action-dvac-adv
BUNDLE=/data/chenyiteng/tmp/fastwam-action-dvac-adv-a6ad77ea.bundle
OLD=01d9d42a0d18df481b660a17aea72830ed4fb147
NEW=a6ad77ea9ee9bf0b355251324c4cf88b6e9a47e7
BRANCH=codex/sz-fastwam-action-dvac-adv
test "$(git -C "$WT" rev-parse HEAD)" = "$OLD"
test -z "$(git -C "$WT" status --porcelain)"
git -C "$WT" fetch "$BUNDLE" "refs/heads/$BRANCH"
test "$(git -C "$WT" rev-parse FETCH_HEAD)" = "$NEW"
git -C "$WT" merge --ff-only FETCH_HEAD
test "$(git -C "$WT" rev-parse HEAD)" = "$NEW"
test -z "$(git -C "$WT" status --porcelain)"
git -C "$WT" push personal "$BRANCH"
printf 'pushed_head=%s\n' "$(git -C "$WT" rev-parse HEAD)"
