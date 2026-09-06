#!/usr/bin/env bash
set -euo pipefail

PGID=1267712
UID_EXPECTED=1003
test "$(id -u)" = "$UID_EXPECTED"
leader_args=$(ps -p "$PGID" -o args=)
case "$leader_args" in
  *"timeout --signal=INT"*"uv pip install"*"d64c4b005459db10c5dd867d8b30a87d5bda9bdb"*) ;;
  *) printf 'unexpected process leader: %s\n' "$leader_args" >&2; exit 1 ;;
esac
test "$(ps -p "$PGID" -o pgid= | tr -d ' ')" = "$PGID"

printf '%s\n' '=== OWNED PROCESS GROUP BEFORE STOP ==='
ps -u "$UID_EXPECTED" -o pid=,ppid=,pgid=,stat=,etime=,args= \
  | awk -v pgid="$PGID" '$3 == pgid'

kill -TERM -- "-$PGID"
for _ in 1 2 3 4 5; do
  if ! kill -0 -- "-$PGID" 2>/dev/null; then
    break
  fi
  sleep 1
done
if kill -0 -- "-$PGID" 2>/dev/null; then
  kill -KILL -- "-$PGID"
fi
! kill -0 -- "-$PGID" 2>/dev/null
printf '%s\n' 'R1_CUROBO_RESOLVER_DRIFT_STOPPED'
