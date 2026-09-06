set -u
runtime=/root/autodl-tmp/experiment_exports/qam_formal_resume100_to380_20260801_v4/runtime
pid=$(cat "$runtime/driver.pid")
cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline")
date '+STOP_REQUEST_TIME=%F %T %Z'
printf 'TARGET_PID=%s\n' "$pid"
printf 'TARGET_CMD=%s\n' "$cmd"
case "$cmd" in
  *qam_formal_resume100_to380_20260801_v4*) ;;
  *) echo TARGET_MISMATCH; exit 42 ;;
esac
grep -a 'Global Step:' "$runtime/driver.log" | tail -n 1
kill -TERM "$pid"
echo TERM_SENT=yes
for _ in $(seq 1 30); do
  if ! kill -0 "$pid" 2>/dev/null; then
    break
  fi
  sleep 2
done
if kill -0 "$pid" 2>/dev/null; then
  echo DRIVER_EXITED=no
else
  echo DRIVER_EXITED=yes
fi
date '+STOP_CHECK_TIME=%F %T %Z'
printf 'EXIT_FILE='
if test -f "$runtime/driver.exit"; then
  cat "$runtime/driver.exit"
else
  echo none
fi
pgrep -af 'qam_formal_resume100_to380_20260801_v4|robotwin_adjust_bottle_qam_formal_resume100_to380_20260801_v4' || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
