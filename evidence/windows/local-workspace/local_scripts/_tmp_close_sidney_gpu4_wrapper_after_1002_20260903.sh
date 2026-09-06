set -eu
pid=2071299
run_root=/data/chenyiteng/results/lerobot-sidney/pi05_robotwin-e49e2ab/move_stapler_pad-official-valid-seed1002-20260903
log="$run_root/eval.log"
test -s "$log"
grep -q "'pc_success': 100.0" "$log"
grep -q "'successes': \[True\]" "$log"
non_zombie=$(ps -o stat= --ppid "$pid" | awk '$1 !~ /^Z/ {print $1}')
if [ -n "$non_zombie" ]; then
  printf 'refuse_live_children pid=%s states=%s\n' "$pid" "$non_zombie"
  ps -o pid,ppid,stat,args --ppid "$pid"
  exit 43
fi
stat=$(ps -o stat= -p "$pid" | tr -d ' ')
case "$stat" in T*|t*) ;; *) printf 'refuse_not_stopped pid=%s stat=%s\n' "$pid" "$stat"; exit 44 ;; esac
kill -KILL "$pid"
printf 'closed_stopped_gpu4_wrapper=%s after_seed1002\n' "$pid"
printf 'WRAPPER_MANUALLY_CLOSED_AFTER_FINAL_METRICS task=move_stapler_pad seed=1002\n' > "$run_root/result.txt"
printf 'Final official metrics are complete; outer loop was closed before queued duplicate seed1004.\n' > "$run_root/manual_close_note.txt"
grep -E "'pc_success':|'successes':|eval_s" "$log" | tail -3
