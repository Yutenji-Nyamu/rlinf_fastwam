#!/usr/bin/env bash
set -euo pipefail

root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-observe-7d07a421
expected_head=f7cf0f6092b92e0ab2b813bc8acd8ee132a288e5
expected_branch=codex/sz-current-pi0-dvac-observe

cd "$root"
head=$(git rev-parse HEAD)
branch=$(git branch --show-current)
status=$(git status --porcelain=v1)
upstream=$(git rev-parse '@{upstream}')

printf 'RLINF_HEAD=%s\n' "$head"
printf 'RLINF_BRANCH=%s\n' "$branch"
printf 'RLINF_UPSTREAM=%s\n' "$upstream"
printf 'RLINF_STATUS=%s\n' "${status:-CLEAN}"
test "$head" = "$expected_head"
test "$branch" = "$expected_branch"
test "$upstream" = "$expected_head"
test -z "$status"

test -x /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
test -d /data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
test -d /data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support

echo 'GPU_RAY_MODEL_SIM_USED=0'
