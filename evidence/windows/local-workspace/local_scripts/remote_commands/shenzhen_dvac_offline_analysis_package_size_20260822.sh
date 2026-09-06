#!/usr/bin/env bash
set -euo pipefail

OUTPUT=/data/chenyiteng/results/dvac-observation/analysis/sz-dvac-pi0-fixed64-fastwam-adjust-p1-v1
test -d "$OUTPUT"
FULL_GZIP_BYTES=$(tar -C "$OUTPUT" -czf - . | wc -c)
COMPACT_GZIP_BYTES=$(tar -C "$OUTPUT" --exclude='./query_horizon.csv' -czf - . | wc -c)
printf 'FULL_TAR_GZ_STREAM_BYTES=%s\n' "$FULL_GZIP_BYTES"
printf 'COMPACT_WITHOUT_QUERY_HORIZON_TAR_GZ_STREAM_BYTES=%s\n' "$COMPACT_GZIP_BYTES"
printf 'PACKAGE_SIZE_ESTIMATE_OK\n'
