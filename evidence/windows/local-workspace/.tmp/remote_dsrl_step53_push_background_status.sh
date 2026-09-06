set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
RUN=$REPO/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1
EXPECTED_HEAD=4447d40211be8c78874bf4b000c871e7fbd93561
LOG=/tmp/dsrl_step53_push_4447d402.log
STATUS=/tmp/dsrl_step53_push_4447d402.status
PIDFILE=/tmp/dsrl_step53_push_4447d402.pid

cd "$REPO"
test "$(git rev-parse HEAD)" = "$EXPECTED_HEAD"
kill -0 "$(cat "$RUN/formal.pid")"

echo "PUSH_PID=$(cat "$PIDFILE")"
if test -s "$STATUS"; then
  echo "PUSH_DONE=1"
  echo "PUSH_STATUS=$(cat "$STATUS")"
else
  echo "PUSH_DONE=0"
  if kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "PUSH_ALIVE=1"
  else
    echo "PUSH_ALIVE=0"
  fi
fi

echo "PUSH_LOG_BEGIN"
tail -n 30 "$LOG"
echo "PUSH_LOG_END"
echo "HEAD=$(git rev-parse HEAD)"
echo "UPSTREAM=$(git rev-parse '@{upstream}')"
echo "STATUS_BEGIN"
git status --short
echo "STATUS_END"
echo "LAST_STEP=$(grep 'Global Step:' "$RUN/formal_driver.log" | tail -n 1)"
