#!/usr/bin/env bash
set -euo pipefail

PARENT=/data/chenyiteng/results/dvac-observation/analysis
NAME=sz-dvac-fastwam-move-stapler-p2-phase-v1
SOURCE="$PARENT/$NAME"
ARCHIVE="$PARENT/$NAME.tar.gz"

test -d "$SOURCE"
test ! -e "$ARCHIVE"
tar -C "$PARENT" -czf "$ARCHIVE" "$NAME"
stat -c 'ARCHIVE_BYTES=%s ARCHIVE=%n' "$ARCHIVE"
sha256sum "$ARCHIVE"
printf '%s\n' 'PHASE_ARCHIVE_OK'
