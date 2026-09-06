set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
RUN=$REPO/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1
EXPECTED_HEAD=51753eab66e49454f7ba5c56020cf49be706aba8

cd "$REPO"
test "$(git branch --show-current)" = codex/dsrl-pi0-robotwin
test "$(git rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git status --porcelain)"
! kill -0 "$(cat "$RUN/formal.pid")" 2>/dev/null

timeout --signal=TERM --kill-after=5s 180s \
  git -c http.version=HTTP/1.1 push personal HEAD:codex/dsrl-pi0-robotwin

test "$(git rev-parse '@{upstream}')" = "$EXPECTED_HEAD"
test -z "$(git status --porcelain)"
echo "HEAD=$EXPECTED_HEAD"
echo "UPSTREAM=$(git rev-parse '@{upstream}')"
echo "STATUS=CLEAN"
echo "DRIVER_ALIVE=0"
