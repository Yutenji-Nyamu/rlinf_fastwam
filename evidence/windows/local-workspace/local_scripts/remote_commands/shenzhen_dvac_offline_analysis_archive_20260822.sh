#!/usr/bin/env bash
set -euo pipefail

OUT=/data/chenyiteng/results/dvac-observation/analysis/sz-dvac-pi0-fixed64-fastwam-adjust-p1-v1
ARCHIVE=/data/chenyiteng/results/dvac-observation/analysis/sz-dvac-pi0-fixed64-fastwam-adjust-p1-v1.tar.gz

test -d "$OUT"
test ! -e "$ARCHIVE"
test "$(find "$OUT" -type f | wc -l)" -eq 35

cd "$(dirname "$OUT")"
tar -czf "$ARCHIVE" "$(basename "$OUT")"

printf 'archive=%s\nbytes=%s\nsha256=%s\n' \
  "$ARCHIVE" \
  "$(stat -c %s "$ARCHIVE")" \
  "$(sha256sum "$ARCHIVE" | awk '{print $1}')"
printf '%s\n' 'SZ_DVAC_OFFLINE_ANALYSIS_ARCHIVE_OK'
