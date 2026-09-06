set -euo pipefail

RUN_ROOT=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_smoke_v1
SCRIPT="$RUN_ROOT/base_freeze_validate.sh"
LOG="$RUN_ROOT/base_freeze_validation.log"
EXIT_FILE="$RUN_ROOT/base_freeze_validation.exit"
PID_FILE="$RUN_ROOT/base_freeze_validation.pid"

test -s "$SCRIPT"
test ! -e "$LOG"
test ! -e "$EXIT_FILE"
test ! -e "$PID_FILE"

nohup bash -c '
  script=$1
  exit_file=$2
  bash "$script"
  code=$?
  printf "%s\n" "$code" > "$exit_file"
  exit "$code"
' _ "$SCRIPT" "$EXIT_FILE" > "$LOG" 2>&1 < /dev/null &
pid=$!
printf "%s\n" "$pid" > "$PID_FILE"
echo "BASE_FREEZE_VALIDATION_PID=$pid"
