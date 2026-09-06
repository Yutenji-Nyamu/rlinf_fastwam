#!/usr/bin/env bash
set -euo pipefail

base=/root/autodl-tmp/experiment_exports/qam_qonly_smoke_20260731_v1
runtime="$base/runtime"
archive="$base/qam_qonly_smoke_runtime_evidence_20260731.tar.gz"

/root/autodl-tmp/RLinf/.venv/bin/python \
  /root/autodl-tmp/qam_resource_summary_20260731.py \
  "$runtime/resources.csv" >"$runtime/resource_summary.json"

tar -C "$base" -czf "$archive" runtime

echo "TIME=$(TZ=Asia/Shanghai date '+%F %T %Z')"
sha256sum "$runtime/resource_summary.json" "$archive"
stat -c 'bytes=%s path=%n' "$runtime/resource_summary.json" "$archive"
cat "$runtime/resource_summary.json"
