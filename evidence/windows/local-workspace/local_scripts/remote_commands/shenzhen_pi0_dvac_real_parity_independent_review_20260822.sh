#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-observe-7d07a421
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
EXPECTED=f7cf0f6092b92e0ab2b813bc8acd8ee132a288e5
TOOL=toolkits/probe_pi0_dvac_real_parity.py

test "$(git -C "$WT" rev-parse HEAD)" = "$EXPECTED"
test "$(git -C "$WT" rev-parse '@{upstream}')" = "$EXPECTED"
test "$(git -C "$WT" branch --show-current)" = codex/sz-current-pi0-dvac-observe
test "$(git -C "$WT" diff --cached --name-only)" = "$TOOL"
test -z "$(git -C "$WT" diff --name-only)"

printf '%s\n' '=== STATUS_AND_STAT ==='
git -C "$WT" status --short
git -C "$WT" diff --cached --check
git -C "$WT" diff --cached --stat
printf '%s\n' '=== STAGED_CONTENT ==='
git -C "$WT" show ":$TOOL"

printf '%s\n' '=== CUDA_HIDDEN_FOCUSED_TESTS ==='
export PYTHONPATH="$WT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=
export NVIDIA_VISIBLE_DEVICES=none
cd "$WT"
"$PY" -m ruff check "$TOOL"
"$PY" -m ruff format --check "$TOOL"
"$PY" -c 'from pathlib import Path; p=Path("toolkits/probe_pi0_dvac_real_parity.py"); compile(p.read_text(encoding="utf-8"), str(p), "exec")'
"$PY" "$TOOL" self-test

test -z "$(git -C "$WT" diff --name-only)"
test "$(git -C "$WT" diff --cached --name-only)" = "$TOOL"
git -C "$WT" diff --cached --check
printf '%s\n' 'PI0_DVAC_REAL_PARITY_INDEPENDENT_REVIEW_CHECKS_OK'
