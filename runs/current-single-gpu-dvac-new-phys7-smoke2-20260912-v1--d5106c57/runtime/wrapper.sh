#!/usr/bin/env bash
set +e
runtime=$1
date --iso-8601=seconds > "$runtime/started_at.txt"
bash "$runtime/command.sh" > "$runtime/driver.log" 2>&1 &
child=$!
printf '%s\n' "$child" > "$runtime/child.pid"
wait "$child"
rc=$?
printf '%s\n' "$rc" > "$runtime/exit_code.txt"
date --iso-8601=seconds > "$runtime/finished_at.txt"
exit "$rc"
