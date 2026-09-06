#!/usr/bin/env bash
set -euo pipefail

PARENT=/data/chenyiteng/results/dvac-observation/analysis
NAME=sz-dvac-pi0-fixed64-fastwam-four-tasks-v1
OUTPUT="$PARENT/$NAME"
ARCHIVE="$PARENT/$NAME.tar.gz"

test -d "$OUTPUT"
test -f "$OUTPUT/analysis_summary.json"
test ! -e "$ARCHIVE"
cd "$PARENT"
tar -czf "$ARCHIVE" "$NAME"
stat -c 'ARCHIVE_BYTES=%s' "$ARCHIVE"
sha256sum "$ARCHIVE"
printf '%s\n' 'SZ_DVAC_ALL_FOUR_TASKS_ARCHIVE_OK'
