#!/usr/bin/env bash
set +e
runtime=$1
cleanup_scorer() {
    rc=$?
    trap - EXIT
    /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -B /data/chenyiteng/results/server-maintenance-20260910/bc-iql-formal/formal.py scorer_stop >> "$runtime/scorer-cleanup.log" 2>&1
    cleanup_rc=$?
    printf '%s\n' "$cleanup_rc" > "$runtime/scorer_cleanup_exit_code.txt"
    if [ "$rc" -eq 0 ] && [ "$cleanup_rc" -ne 0 ]; then rc=92; fi
    printf '%s\n' "$rc" > "$runtime/exit_code.txt"
    date --iso-8601=seconds > "$runtime/finished_at.txt"
    exit "$rc"
}
trap cleanup_scorer EXIT
trap 'if [ -n "${child:-}" ]; then kill -TERM "$child" 2>/dev/null; wait "$child"; fi; exit 143' TERM
trap 'if [ -n "${child:-}" ]; then kill -TERM "$child" 2>/dev/null; wait "$child"; fi; exit 130' INT
ulimit -n 4096 || exit 91
date --iso-8601=seconds > "$runtime/started_at.txt"
timeout --signal=TERM --kill-after=180s 172800s bash "$runtime/command.sh" > "$runtime/driver.log" 2>&1 &
child=$!
printf '%s\n' "$child" > "$runtime/timeout.pid"
wait "$child"
rc=$?
printf '%s\n' "$rc" > "$runtime/training_exit_code.txt"
exit "$rc"
