#!/usr/bin/env bash
set -euo pipefail

root=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711
cd "$root"

echo "HEAD=$(git rev-parse HEAD)"
echo "BRANCH=$(git branch --show-current)"
echo 'STATUS_BEGIN'
git status --short
echo 'STATUS_END'
echo 'STAGED_NAMES_BEGIN'
git diff --cached --name-only
echo 'STAGED_NAMES_END'
grep -nE 'reset.seed|reset_seed|reset-seed|4300000|4300001|play_once|take_action' \
  experiments/robotwin/fastwam_real_query_parity.py
git diff --cached --check
echo 'GPU_RAY_MODEL_SIM_USED=0'
