#!/usr/bin/env bash
set -euo pipefail

venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
src=/data/chenyiteng/results/rlinf-shared-ray/formal-rlt-dsrl-20260823-v1
dst=/data/chenyiteng/results/rlinf-shared-ray/formal-rlt-dsrl-20260823-v1-failed-public-ip
rlt_chain=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage1-2k-stage2-8env250-20260823-v1
next_tmp=/data/chenyiteng/ray/rlt-dsrl-v2

test "$(readlink -m "$src")" = "$src"
test "$(readlink -m "$dst")" = "$dst"
test -d "$src"
test -s "$src/ray_start.log"
# Ray colorizes the IP in this log, so match the address rather than a
# contiguous colorized sentence.
grep -F '120.241.223.9' "$src/ray_start.log" >/dev/null
test ! -e "$dst"
test ! -e "$rlt_chain"
test ! -e "$next_tmp"
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[[:space:]]*[0-9]+'; then
  printf '%s\n' 'a GPU compute process exists; refusing Ray cleanup' >&2
  exit 1
fi

# The failed head has already exited in some runs; `ray stop` then returns
# non-zero even though there is nothing left to stop.  The process assertions
# below are the authoritative safety check.
"$venv/bin/ray" stop --force || true
for _ in $(seq 1 30); do
  if ! pgrep -u "$(id -u)" -x raylet >/dev/null && ! pgrep -u "$(id -u)" -x gcs_server >/dev/null; then
    break
  fi
  sleep 1
done
test -z "$(pgrep -u "$(id -u)" -x raylet || true)"
test -z "$(pgrep -u "$(id -u)" -x gcs_server || true)"
mv -- "$src" "$dst"
printf 'preserved_failed_attempt=%s\n' "$dst"
printf 'next_node_ip=127.0.0.1\nnext_short_temp=%s\n' "$next_tmp"
