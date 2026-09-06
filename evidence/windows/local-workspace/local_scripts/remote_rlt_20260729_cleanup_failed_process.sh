#!/usr/bin/env bash
set -euo pipefail

FAILED=/root/autodl-tmp/datasets/robotwin2/intermediate/9dc9299c163db059931898a9f0852098a61155a1/adjust_bottle/.pi0-aloha-clean50-v1.process.AdWcXR
TARGET=/root/autodl-tmp/datasets/robotwin2/intermediate/9dc9299c163db059931898a9f0852098a61155a1/adjust_bottle/pi0-aloha-clean50-v1
EXPECTED_PARENT=/root/autodl-tmp/datasets/robotwin2/intermediate/9dc9299c163db059931898a9f0852098a61155a1/adjust_bottle

test ! -e "$TARGET"
test -d "$FAILED"
test "$(dirname "$(realpath -e -- "$FAILED")")" = "$EXPECTED_PARENT"
case "$(basename "$FAILED")" in
  .pi0-aloha-clean50-v1.process.*) ;;
  *) exit 31 ;;
esac
du -sh -- "$FAILED"
rm -rf -- "$FAILED"
test ! -e "$FAILED"
printf 'REMOVED_FAILED_STAGING\t%s\n' "$FAILED"
