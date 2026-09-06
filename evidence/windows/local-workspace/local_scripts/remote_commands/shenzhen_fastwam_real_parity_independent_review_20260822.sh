#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711
PY=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128/bin/python
EXPECTED=fc652fb49cd32350eca15734b5c7124c0b8c2c02

test "$(git -C "$WT" rev-parse HEAD)" = "$EXPECTED"
test "$(git -C "$WT" branch --show-current)" = codex/sz-fastwam-dvac-observe
test "$(git -C "$WT" diff --cached --name-only | wc -l)" -eq 3
test -z "$(git -C "$WT" diff --name-only)"

printf '%s\n' '=== STATUS_AND_STAT ==='
git -C "$WT" status --short
git -C "$WT" diff --cached --check
git -C "$WT" diff --cached --stat

printf '%s\n' '=== MODEL_STAGED_DIFF ==='
git -C "$WT" diff --cached -- src/fastwam/models/wan22/fastwam.py
printf '%s\n' '=== HARNESS_STAGED_CONTENT ==='
git -C "$WT" show :experiments/robotwin/fastwam_real_query_parity.py
printf '%s\n' '=== TEST_STAGED_CONTENT ==='
git -C "$WT" show :tests/test_fastwam_real_query_parity.py

printf '%s\n' '=== CUDA_HIDDEN_FOCUSED_TESTS ==='
export CUDA_VISIBLE_DEVICES=
export PYTHONDONTWRITEBYTECODE=1
cd "$WT"
"$PY" -m unittest -v tests/test_fastwam_real_query_parity.py
"$PY" experiments/robotwin/fastwam_real_query_parity.py --help >/dev/null

test -z "$(git -C "$WT" diff --name-only)"
test "$(git -C "$WT" diff --cached --name-only | wc -l)" -eq 3
git -C "$WT" diff --cached --check
printf '%s\n' 'FASTWAM_REAL_PARITY_INDEPENDENT_REVIEW_CHECKS_OK'
