#!/usr/bin/env bash
set -euo pipefail

pid=1352354
expected='evaluations/eval_embodied_agent.py'
run_name='ppo-reload-fixed64-4gpu4567-v1'

if ! kill -0 "$pid" 2>/dev/null; then
  printf 'owned reload driver already exited: pid=%s\n' "$pid"
  exit 0
fi

cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline")
case "$cmd" in
  *"$expected"*"$run_name"*) ;;
  *)
    printf 'refusing to signal unexpected pid=%s cmd=%s\n' "$pid" "$cmd" >&2
    exit 1
    ;;
esac

printf 'sending SIGINT to owned reload driver pid=%s\n' "$pid"
kill -INT "$pid"
for _ in $(seq 1 30); do
  kill -0 "$pid" 2>/dev/null || {
    printf '%s\n' 'owned reload driver exited after SIGINT'
    exit 0
  }
  sleep 1
done

printf 'sending SIGTERM to owned reload driver pid=%s after 30s\n' "$pid"
kill -TERM "$pid"
