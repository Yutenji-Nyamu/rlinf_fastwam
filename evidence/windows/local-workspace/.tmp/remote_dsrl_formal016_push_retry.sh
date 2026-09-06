set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
RUN=$REPO/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1
EXPECTED_PREFIX=48a775db

cd "$REPO"
test "$(git branch --show-current)" = codex/dsrl-pi0-robotwin
expected_head=$(git rev-parse --verify "$EXPECTED_PREFIX^{commit}")
test "$(git rev-parse HEAD)" = "$expected_head"
test -z "$(git status --porcelain)"
! kill -0 "$(cat "$RUN/formal.pid")" 2>/dev/null
timeout --signal=TERM --kill-after=5s 180s \
  git -c http.version=HTTP/1.1 push personal HEAD:codex/dsrl-pi0-robotwin
test "$(git rev-parse '@{upstream}')" = "$expected_head"
echo "HEAD=$expected_head"
echo "UPSTREAM=$(git rev-parse '@{upstream}')"
echo "STATUS=CLEAN"
echo "DRIVER_ALIVE=0"
