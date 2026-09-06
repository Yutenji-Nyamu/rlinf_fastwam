#!/usr/bin/env bash
set -euo pipefail

worktree=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
branch=codex/rlt-pi0-robotwin
ledger=docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md

cd "$worktree"
test "$(git branch --show-current)" = "$branch"
test "$(git status --short)" = " M $ledger"
git diff --check -- "$ledger"
git add -- "$ledger"
git diff --cached --check
git commit -m "docs(rlt): close clean50 operation ledger"

commit=$(git rev-parse HEAD)
printf 'COMMIT %s\n' "$commit"
timeout 30s git push personal "$branch"
printf '%s\n' 'PUSH_OK'
git status --short
git rev-list --left-right --count HEAD...@{upstream}
