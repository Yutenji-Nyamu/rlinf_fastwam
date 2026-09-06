#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
upload=/root/autodl-tmp/tmp/rlt_formal250_upload_20260730
path=tests/unit_tests/test_robotwin_seed_partition.py
cd "$repo"

test "$(sha256sum "$path" | awk '{print $1}')" = \
  e4b5686e5d4bc68961cf7238d7b42be96f724e6bc948c530e2ddb4f4c8d3c093
test "$(sha256sum "$upload/$path" | awk '{print $1}')" = \
  4d764fc0a0e042e1b71a93d98d5e9a2627e6ed8a899c9dee8a92528303ad3c0f
install -m 0644 "$upload/$path" "$path"
git diff --check
sha256sum "$path"
