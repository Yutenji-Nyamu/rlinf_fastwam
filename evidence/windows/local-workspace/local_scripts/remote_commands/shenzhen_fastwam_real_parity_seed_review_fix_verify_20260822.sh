#!/usr/bin/env bash
set -euo pipefail

root=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711
target=experiments/robotwin/fastwam_real_query_parity.py
cd "$root"

test "$(git rev-parse HEAD)" = fc652fb49cd32350eca15734b5c7124c0b8c2c02
test "$(git branch --show-current)" = codex/sz-fastwam-dvac-observe
test -z "$(git diff --name-only)"
test "$(git diff --cached --name-only | wc -l)" -eq 3
test "$(grep -c 'parser.add_argument("--reset-seed", type=int, default=4300001)' "$target")" -eq 1
test "$(grep -c '4300000' "$target")" -eq 0
test "$(grep -c 'task_env.play_once()' "$target")" -eq 1
git diff --cached --check
git diff --cached --stat
echo 'DEFAULT_RESET_SEED=4300001'
echo 'OFFICIAL_EXPERT_PLAY_ONCE_COUNT=1'
echo 'UNSTAGED_DIFF=0'
echo 'COMMIT_PUSH_GPU_RAY_MODEL_SIM_USED=0'
