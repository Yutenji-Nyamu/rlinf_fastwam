#!/usr/bin/env bash
set -euo pipefail
WORKTREE=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421
git -C "$WORKTREE" apply -
git -C "$WORKTREE" diff --check
git -C "$WORKTREE" status --short
