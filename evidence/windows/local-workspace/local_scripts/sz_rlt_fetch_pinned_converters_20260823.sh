#!/usr/bin/env bash
set -euo pipefail

PIN=c3ddfa8b97d5519efa828b075999bd0006778e5e
ROOT=/data/chenyiteng/datasets/robotwin2/tooling/RoboTwin-$PIN
PROXY=http://127.0.0.1:7890

fetch() {
  rel=$1
  expected=$2
  target=$ROOT/$rel
  test ! -e "$target"
  mkdir -p "$(dirname "$target")"
  curl --proxy "$PROXY" --retry 6 --retry-delay 1 --retry-all-errors \
    -fsSL --connect-timeout 10 --max-time 120 \
    "https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/$PIN/$rel" \
    -o "$target"
  test "$(sha256sum "$target" | awk '{print $1}')" = "$expected"
  echo "FETCHED $rel"
}

fetch policy/pi0/scripts/process_data.py \
  b462918bf3f41f6d2fc30c3498381ac3cc7d8ce7a8bd6333fafb925e7d9d5590
fetch policy/pi0/examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py \
  b8f0829329e099b7246b3d6467cec3ea4d60767eedd219b825d0b7f26bb7c373

echo "SUCCESS $(date --iso-8601=seconds)"
