#!/usr/bin/env bash
set -euo pipefail

root=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711
target=experiments/robotwin/fastwam_real_query_parity.py
cd "$root"

test "$(git rev-parse HEAD)" = fc652fb49cd32350eca15734b5c7124c0b8c2c02
test "$(git branch --show-current)" = codex/sz-fastwam-dvac-observe
test -z "$(git diff --name-only)"
test "$(git diff --cached --name-only | wc -l)" -eq 3
test "$(grep -c 'parser.add_argument("--reset-seed", type=int, default=4300000)' "$target")" -eq 1
test "$(grep -c 'task_env.play_once()' "$target")" -eq 1
echo 'FASTWAM_REAL_PARITY_SEED_FIX_UPLOAD_READY=1'
echo 'GPU_RAY_MODEL_SIM_USED=0'
