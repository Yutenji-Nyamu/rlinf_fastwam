#!/usr/bin/env bash
set -euo pipefail

target=/root/autodl-tmp/datasets/robotwin2/source/9dc9299c163db059931898a9f0852098a61155a1/dataset/adjust_bottle/aloha-agilex_clean_50.zip
script=/root/autodl-tmp/tmp/rlt_clean50_download_20260729_v2.sh
log=/root/autodl-tmp/tmp/rlt_clean50_download_20260729_v2.log

echo "DIAGNOSE_TIME $(date -Is)"
set +e
set -o pipefail
unzip -Z1 "$target" | head -30 >/dev/null
legacy_pipeline_rc=$?
set -e
echo "legacy_unzip_head_pipeline_rc=$legacy_pipeline_rc"

if [[ -e "$log" ]]; then
  echo "FAIL: v2 evidence log already exists" >&2
  exit 40
fi

chmod 700 "$script"
if bash "$script" >"$log" 2>&1; then
  producer_v2_rc=0
else
  producer_v2_rc=$?
fi
echo "producer_v2_rc=$producer_v2_rc"
cat "$log"
if [[ "$producer_v2_rc" -ne 0 ]]; then
  exit "$producer_v2_rc"
fi
