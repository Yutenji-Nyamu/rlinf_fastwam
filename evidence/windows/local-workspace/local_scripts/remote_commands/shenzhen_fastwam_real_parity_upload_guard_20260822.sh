#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711
EXPECTED=fc652fb49cd32350eca15734b5c7124c0b8c2c02

test "$(id -un)" = chenyiteng
test "$(git -C "$WT" rev-parse HEAD)" = "$EXPECTED"
test -z "$(git -C "$WT" status --porcelain=v1 --untracked-files=all)"
printf '%s  %s\n' \
  230a6c8257b692ccb7f629a19da2d17e17a8d7b24bd66d9bd4fabd2a353738fa \
  "$WT/src/fastwam/models/wan22/fastwam.py" | sha256sum -c -
test ! -e "$WT/experiments/robotwin/fastwam_real_query_parity.py"
test ! -e "$WT/tests/test_fastwam_real_query_parity.py"
printf 'FASTWAM_REAL_PARITY_UPLOAD_GUARD_OK\n'
