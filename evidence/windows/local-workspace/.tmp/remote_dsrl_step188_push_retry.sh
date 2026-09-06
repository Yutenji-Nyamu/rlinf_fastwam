set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
RUN=$REPO/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1

cd "$REPO"
test "$(git branch --show-current)" = codex/dsrl-pi0-robotwin
test -z "$(git status --porcelain)"
kill -0 "$(cat "$RUN/formal.pid")"

set +e
timeout --signal=TERM --kill-after=5s 55s \
  git -c http.version=HTTP/1.1 push personal HEAD:codex/dsrl-pi0-robotwin
status=$?
set -e

echo "PUSH_STATUS=$status"
echo "HEAD=$(git rev-parse HEAD)"
echo "UPSTREAM=$(git rev-parse '@{upstream}')"
echo "STATUS_BEGIN"
git status --short
echo "STATUS_END"
kill -0 "$(cat "$RUN/formal.pid")"
echo "DRIVER_ALIVE=1"
echo "LAST_STEP=$(grep 'Global Step:' "$RUN/formal_driver.log" | tail -n 1)"
exit "$status"
