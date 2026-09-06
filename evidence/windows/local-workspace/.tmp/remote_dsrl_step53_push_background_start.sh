set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
RUN=$REPO/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1
EXPECTED_HEAD=4447d40211be8c78874bf4b000c871e7fbd93561
LOG=/tmp/dsrl_step53_push_4447d402.log
STATUS=/tmp/dsrl_step53_push_4447d402.status
PIDFILE=/tmp/dsrl_step53_push_4447d402.pid

cd "$REPO"
test "$(git rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git status --porcelain)"
kill -0 "$(cat "$RUN/formal.pid")"
test ! -e "$LOG"
test ! -e "$STATUS"
test ! -e "$PIDFILE"

nohup bash -lc '
  cd /root/autodl-tmp/RLinf_fastwam_rlinf
  set +e
  GIT_TERMINAL_PROMPT=0 timeout --signal=TERM --kill-after=5s 180s \
    git -c http.version=HTTP/1.1 push personal HEAD:codex/dsrl-pi0-robotwin
  code=$?
  printf "%s\n" "$code" > /tmp/dsrl_step53_push_4447d402.status
' >"$LOG" 2>&1 < /dev/null &
echo "$!" > "$PIDFILE"

echo "BACKGROUND_PUSH_STARTED=1"
echo "PUSH_PID=$(cat "$PIDFILE")"
echo "TRAIN_DRIVER_ALIVE=1"
