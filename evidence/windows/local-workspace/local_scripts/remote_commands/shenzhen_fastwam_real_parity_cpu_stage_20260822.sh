#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711
PY=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128/bin/python
EXPECTED=fc652fb49cd32350eca15734b5c7124c0b8c2c02

test "$(git -C "$WT" rev-parse HEAD)" = "$EXPECTED"
test "$(git -C "$WT" branch --show-current)" = codex/sz-fastwam-dvac-observe
test -f "$WT/experiments/robotwin/fastwam_real_query_parity.py"
test -f "$WT/tests/test_fastwam_real_query_parity.py"

export CUDA_VISIBLE_DEVICES=
cd "$WT"
"$PY" -m py_compile \
  src/fastwam/models/wan22/fastwam.py \
  experiments/robotwin/fastwam_real_query_parity.py \
  tests/test_fastwam_real_query_parity.py
"$PY" -m unittest -v tests/test_fastwam_real_query_parity.py
"$PY" experiments/robotwin/fastwam_real_query_parity.py --help >/dev/null

git add \
  src/fastwam/models/wan22/fastwam.py \
  experiments/robotwin/fastwam_real_query_parity.py
git add -f tests/test_fastwam_real_query_parity.py
git diff --cached --check
test "$(git diff --cached --name-only | wc -l)" -eq 3
git diff --cached --stat
git status --short
printf 'FASTWAM_REAL_PARITY_CPU_STAGE_OK\n'
