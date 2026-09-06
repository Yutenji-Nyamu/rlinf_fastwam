#!/usr/bin/env bash
set -euo pipefail

upload_root=/root/autodl-tmp/tmp/rlt_formal250_upload_20260730
mkdir -p \
  "$upload_root/examples/embodiment/config" \
  "$upload_root/rlinf/envs/robotwin/seeds" \
  "$upload_root/tests/unit_tests"
printf 'upload_root=%s\n' "$upload_root"
