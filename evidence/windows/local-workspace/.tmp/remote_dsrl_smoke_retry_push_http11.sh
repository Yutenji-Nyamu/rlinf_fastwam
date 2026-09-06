set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
BRANCH=codex/dsrl-pi0-robotwin
COMMIT=ff0d8d2270aa5ec4f0934c997f53118460bf8152

test "$(git -C "$REPO" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
timeout --signal=TERM --kill-after=5s 60s \
  git -C "$REPO" \
    -c http.version=HTTP/1.1 \
    -c http.lowSpeedLimit=1 \
    -c http.lowSpeedTime=30 \
    push personal "$BRANCH"
test "$(git -C "$REPO" rev-parse '@{upstream}')" = "$COMMIT"
echo "PUSHED_HEAD=$COMMIT"
