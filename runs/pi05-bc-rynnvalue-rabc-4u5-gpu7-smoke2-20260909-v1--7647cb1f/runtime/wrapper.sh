#!/usr/bin/env bash
set +e
runtime=$1
ulimit -n 4096 || exit 91
date --iso-8601=seconds > "$runtime/started_at.txt"
timeout --signal=TERM --kill-after=180s 3600s bash "$runtime/command.sh" > "$runtime/driver.log" 2>&1 &
child=$!
printf '%s\n' "$child" > "$runtime/timeout.pid"
wait "$child"
rc=$?
printf '%s\n' "$rc" > "$runtime/exit_code.txt"
date --iso-8601=seconds > "$runtime/finished_at.txt"
exit "$rc"
