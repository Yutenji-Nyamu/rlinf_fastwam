#!/usr/bin/env bash
set -euo pipefail

failed=/data/chenyiteng/results/rlinf-rlt/smoke-stage1-current-ar-2step-20260823
archive=/data/chenyiteng/results/rlinf-rlt/smoke-stage1-current-ar-2step-20260823-attempt1-precompose-failed

test -d "$failed"
test ! -e "$failed/runtime/wrapper.pid"
test ! -e "$archive"

mv "$failed" "$archive"
find "$archive" -maxdepth 3 -printf '%P %y %s\n' | sort
printf '%s\n' 'RLT_STAGE1_ATTEMPT1_ARCHIVED'
