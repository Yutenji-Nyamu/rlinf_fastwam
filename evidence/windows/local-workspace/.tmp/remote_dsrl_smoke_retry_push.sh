set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
BRANCH=codex/dsrl-pi0-robotwin
HEAD=ff0d8d22

test "$(git -C "$REPO" branch --show-current)" = "$BRANCH"
test "$(git -C "$REPO" rev-parse --short=8 HEAD)" = "$HEAD"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
git -C "$REPO" push personal "$BRANCH"
commit=$(git -C "$REPO" rev-parse HEAD)
test "$(git -C "$REPO" rev-parse '@{upstream}')" = "$commit"
echo "PUSHED_HEAD=$commit"
