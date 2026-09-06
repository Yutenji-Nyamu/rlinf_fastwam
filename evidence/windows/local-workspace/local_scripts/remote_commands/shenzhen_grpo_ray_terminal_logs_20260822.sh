#!/usr/bin/env bash
set -u

SESSION=/tmp/ray/session_2026-08-22_13-44-12_554736_641978

date --iso-8601=seconds
id
printf 'session=%s\n' "$SESSION"
for f in \
  "$SESSION/logs/gcs_server.out" \
  "$SESSION/logs/raylet.out" \
  "$SESSION/logs/monitor.log" \
  "$SESSION/logs/dashboard.log"; do
  [ -f "$f" ] || { printf 'absent=%s\n' "$f"; continue; }
  echo "=== $f ==="
  stat -c 'bytes=%s mtime=%y path=%n' "$f"
  grep -aEin 'SIGTERM|SIGINT|signal|shutdown|shutting down|killed|kill|OOM|out of memory|fatal|error|failed|GCS|ray stop|connection' "$f" 2>/dev/null | tail -n 120 || true
  echo '--- tail150 ---'
  tail -n 150 "$f" 2>/dev/null || true
done

echo 'SZ_GRPO_RAY_TERMINAL_LOGS_DONE'
