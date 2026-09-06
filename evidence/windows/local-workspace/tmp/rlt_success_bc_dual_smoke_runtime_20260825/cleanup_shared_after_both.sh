#!/usr/bin/env bash
set +e
control_runtime="$1"
method_runtime="$2"
head_pgid="$3"
while [ ! -f "$control_runtime/finished_at.txt" ] || [ ! -f "$method_runtime/finished_at.txt" ]; do
  sleep 2
done
kill -INT -- "-$head_pgid" 2>/dev/null || true
sleep 8
kill -TERM -- "-$head_pgid" 2>/dev/null || true
