#!/usr/bin/env bash
set -euo pipefail

root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-observe-7d07a421
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
tool=toolkits/probe_pi0_dvac_real_parity.py

cd "$root"
test "$(git rev-parse HEAD)" = f7cf0f6092b92e0ab2b813bc8acd8ee132a288e5
test "$(git branch --show-current)" = codex/sz-current-pi0-dvac-observe
test -z "$(git diff --name-only --diff-filter=U)"
cached=$(git diff --cached --name-only)
test -z "$cached" || test "$cached" = "$tool"
test -f "$tool"

export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=''
export NVIDIA_VISIBLE_DEVICES=none

"$venv/bin/python" -m ruff format "$tool"
"$venv/bin/python" -m ruff check "$tool"
"$venv/bin/python" -m ruff format --check "$tool"
"$venv/bin/python" -m py_compile "$tool"
"$venv/bin/python" "$tool" self-test

git add -- "$tool"
test "$(git diff --cached --name-only)" = "$tool"
git diff --cached --check
git diff --cached --stat
git status --short
echo 'GPU_RAY_MODEL_SIM_USED=0'
