#!/usr/bin/env bash
set -euo pipefail

FAILED=/root/autodl-tmp/datasets/robotwin2/raw/9dc9299c163db059931898a9f0852098a61155a1/adjust_bottle/.clean50-v1.extract.kOi5x9
TARGET=/root/autodl-tmp/datasets/robotwin2/raw/9dc9299c163db059931898a9f0852098a61155a1/adjust_bottle/clean50-v1
EXPECTED_PARENT=/root/autodl-tmp/datasets/robotwin2/raw/9dc9299c163db059931898a9f0852098a61155a1/adjust_bottle

test ! -e "$TARGET"
test -d "$FAILED"
test "$(dirname "$(realpath -e -- "$FAILED")")" = "$EXPECTED_PARENT"
case "$(basename "$FAILED")" in
  .clean50-v1.extract.*) ;;
  *) exit 30 ;;
esac
du -sh -- "$FAILED"
rm -rf -- "$FAILED"
test ! -e "$FAILED"
printf 'REMOVED_FAILED_STAGING\t%s\n' "$FAILED"
