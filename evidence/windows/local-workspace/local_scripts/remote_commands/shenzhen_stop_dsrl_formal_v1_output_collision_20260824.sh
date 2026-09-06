#!/usr/bin/env bash
set -euo pipefail

rlt_root=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage1-2k-stage2-8env250-20260824-v2
dsrl_root=/data/chenyiteng/results/rlinf-current-dsrl/formal-dsrl-pi0-robotwin-2gpu4env-200c-20260824-v1
log=$dsrl_root/run

test -d "$dsrl_root"
test -s "$log/owned.pgid"
test -s "$rlt_root/runtime/wrapper.pid"

pgid=$(cat "$log/owned.pgid")
rlt_pid=$(cat "$rlt_root/runtime/wrapper.pid")
case "$pgid" in (*[!0-9]*|'') exit 2;; esac
case "$rlt_pid" in (*[!0-9]*|'') exit 2;; esac
kill -0 "$rlt_pid"

printf '%s\n' \
  'reason=run-scoped RoboTwin train/eval save paths were missing; external Ray workers resolved ./data to /home/chenyiteng/data' \
  "stopped_at=$(date --iso-8601=seconds)" \
  'scope=owned DSRL v1 process group only; RLT Stage1 and shared Ray preserved' \
  > "$log/manual_stop.txt"

if kill -0 "$pgid" 2>/dev/null; then
  kill -TERM -- "-$pgid"
fi

for _ in $(seq 1 90); do
  if ! kill -0 "$pgid" 2>/dev/null; then
    break
  fi
  sleep 2
done

if kill -0 "$pgid" 2>/dev/null; then
  printf '%s\n' "DSRL owned process group $pgid did not stop after 180 seconds" >&2
  exit 3
fi

observer_pid=$(cat "$log/resource_observer.pid" 2>/dev/null || true)
if [[ "$observer_pid" =~ ^[0-9]+$ ]] && kill -0 "$observer_pid" 2>/dev/null; then
  kill -TERM -- "-$observer_pid" 2>/dev/null || kill -TERM "$observer_pid" 2>/dev/null || true
fi

kill -0 "$rlt_pid"
printf 'dsrl_v1_pgid=%s state=stopped\nrlt_pid=%s state=alive\n' "$pgid" "$rlt_pid"
printf '%s\n' 'SZ_DSRL_V1_STOPPED_FOR_OUTPUT_ISOLATION'
