set -euo pipefail

RUN_ROOT=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_smoke_v1
MANIFEST="$RUN_ROOT/ckpt1_before_resume.sha256"
LOG="$RUN_ROOT/ckpt1_verify_after_resume.log"
EXIT_FILE="$RUN_ROOT/ckpt1_verify_after_resume.exit"
PID_FILE="$RUN_ROOT/ckpt1_verify_after_resume.pid"

test -s "$MANIFEST"
test ! -e "$LOG"
test ! -e "$EXIT_FILE"
test ! -e "$PID_FILE"

nohup bash -c '
  manifest=$1
  exit_file=$2
  sha256sum -c "$manifest"
  code=$?
  printf "%s\n" "$code" > "$exit_file"
  exit "$code"
' _ "$MANIFEST" "$EXIT_FILE" > "$LOG" 2>&1 < /dev/null &
pid=$!
printf "%s\n" "$pid" > "$PID_FILE"
echo "CKPT1_VERIFY_PID=$pid"
