#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current
cd "$WT"
git remote -v
git push -u personal codex/sz-current-pi0-dvac-grpo-w0to5
git status --short
echo SZ_W0TO5_PUSH_OK
