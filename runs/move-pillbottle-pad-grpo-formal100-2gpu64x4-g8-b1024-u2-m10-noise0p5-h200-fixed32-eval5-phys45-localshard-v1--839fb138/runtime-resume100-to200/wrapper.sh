#!/usr/bin/env bash
set +e
runtime=$1; command_file=$2
date --iso-8601=seconds > "$runtime/started_at.txt"
timeout --signal=TERM --kill-after=180s 345600s bash -lc "$(cat "$command_file")" > "$runtime/driver.log" 2>&1
rc=$?; printf '%s\n' "$rc" > "$runtime/exit_code.txt"; date --iso-8601=seconds > "$runtime/finished_at.txt"; exit "$rc"
