set -euo pipefail
runtime=/root/autodl-tmp/experiment_exports/qam_formal_20260801_v2/runtime
driver=$(tr -d '[:space:]' < "$runtime/driver.pid")
cmd=$(ps -p "$driver" -o args=)
pgid=$(ps -p "$driver" -o pgid= | tr -d '[:space:]')
test "$driver" = "$pgid"
case "$cmd" in
  *train_embodied_agent.py*) ;;
  *) printf 'refuse_non_training_driver=%s cmd=%s\n' "$driver" "$cmd" >&2; exit 1 ;;
esac
case "$cmd" in
  *qam_formal_20260801_v2*) ;;
  *) printf 'refuse_wrong_run_driver=%s cmd=%s\n' "$driver" "$cmd" >&2; exit 1 ;;
esac
printf 'stop_time=%s driver=%s pgid=%s\n' "$(date --iso-8601=seconds)" "$driver" "$pgid"
kill -TERM -- "-$pgid"
for _ in $(seq 1 30); do
  if ! kill -0 "$driver" 2>/dev/null; then
    break
  fi
  sleep 1
done
printf 'remaining\n'
ps -eo pid,ppid,pgid,stat,cmd | grep -E 'qam_formal_20260801_v2|train_embodied_agent.py|raylet|gcs_server' | grep -v grep || true
printf 'exit_files\n'
for file in "$runtime/exit_code.txt" "$runtime/monitor_exit_code.txt"; do
  if test -f "$file"; then
    printf '%s=%s\n' "$(basename "$file")" "$(tr -d '[:space:]' < "$file")"
  fi
done
