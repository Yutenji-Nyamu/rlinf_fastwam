#!/usr/bin/env bash
set -u

LOGS=/tmp/ray/session_2026-08-22_13-44-12_554736_641978/logs
date --iso-8601=seconds
id
for name in gcs_server.out raylet.out monitor.log; do
  f="$LOGS/$name"
  echo "=== $name SIGNAL_LINES ==="
  grep -aEin 'SIGTERM|SIGINT|SIGKILL|received signal|signal [0-9]+|shutdown|shutting down|ray stop|terminate|termination|exiting|exit code|killed|out of memory|OOM|fatal' "$f" 2>/dev/null | tail -n 60 || true
  echo "=== $name LAST30 ==="
  tail -n 30 "$f" 2>/dev/null || true
done
echo 'SZ_GRPO_RAY_SIGNAL_LINES_DONE'
