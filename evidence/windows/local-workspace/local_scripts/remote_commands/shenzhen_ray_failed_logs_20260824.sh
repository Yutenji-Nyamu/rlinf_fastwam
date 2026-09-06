#!/usr/bin/env bash
set -u

for base in \
  /data/chenyiteng/ray/rlt-dsrl-v1 \
  /data/chenyiteng/ray/rlt-dsrl-v2; do
  printf '\n=== %s ===\n' "$base"
  if [ ! -e "$base" ]; then
    printf '%s\n' absent
    continue
  fi
  readlink -f "$base/session_latest" || true
  logs="$base/session_latest/logs"
  for file in gcs_server.err gcs_server.out raylet.err raylet.out monitor.err monitor.log; do
    if [ -s "$logs/$file" ]; then
      printf '%s\n' "--- $file (tail) ---"
      tail -n 60 "$logs/$file"
    fi
  done
done

printf '\n=== preserved start logs ===\n'
for file in \
  /data/chenyiteng/results/rlinf-shared-ray/formal-rlt-dsrl-20260823-v1-failed-public-ip/ray_start.log \
  /data/chenyiteng/results/rlinf-shared-ray/formal-rlt-dsrl-20260823-v1/ray_start.log; do
  if [ -s "$file" ]; then
    printf '%s\n' "--- $file ---"
    sed -n '1,35p' "$file"
  fi
done
