set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
RUN=$REPO/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1
EXPECTED_HEAD=4447d40211be8c78874bf4b000c871e7fbd93561

cd "$REPO"
test "$(git branch --show-current)" = codex/dsrl-pi0-robotwin
test "$(git rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git status --porcelain)"
kill -0 "$(cat "$RUN/formal.pid")"

set +e
GIT_TERMINAL_PROMPT=0 timeout --signal=TERM --kill-after=5s 55s \
  git -c http.version=HTTP/1.1 push personal HEAD:codex/dsrl-pi0-robotwin
push_status=$?
set -e
echo "PUSH_STATUS=$push_status"

test "$push_status" -eq 0
test "$(git rev-parse '@{upstream}')" = "$EXPECTED_HEAD"
test -z "$(git status --porcelain)"
kill -0 "$(cat "$RUN/formal.pid")"

echo "PUSH_RETRY=PASS"
echo "HEAD=$(git rev-parse HEAD)"
echo "UPSTREAM=$(git rev-parse '@{upstream}')"
echo "LAST_STEP=$(grep 'Global Step:' "$RUN/formal_driver.log" | tail -n 1)"
