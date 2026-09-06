#!/usr/bin/env bash
set -euo pipefail

src=/data/chenyiteng/results/rlinf-shared-ray/formal-rlt-dsrl-20260823-v1
dst=/data/chenyiteng/results/rlinf-shared-ray/formal-rlt-dsrl-20260823-v1-failed-af-unix-path
short_tmp=/data/chenyiteng/ray/rlt-dsrl-v1

test "$(readlink -m "$src")" = "$src"
test "$(readlink -m "$dst")" = "$dst"
test -d "$src"
test -s "$src/ray_start.log"
grep -F 'AF_UNIX path length cannot exceed 107 bytes' "$src/ray_start.log" >/dev/null
test ! -e "$dst"
test ! -e "$short_tmp"
test -z "$(pgrep -u "$(id -u)" -x raylet || true)"
test -z "$(pgrep -u "$(id -u)" -x gcs_server || true)"
mv -- "$src" "$dst"
printf 'preserved_failed_attempt=%s\n' "$dst"
printf 'next_short_temp=%s\n' "$short_tmp"
