#!/usr/bin/env bash
set -euo pipefail

worktree=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
branch=codex/rlt-pi0-robotwin

cd "$worktree"
test "$(git branch --show-current)" = "$branch"

actual_paths=$(git status --short | sed 's/^...//' | LC_ALL=C sort)
expected_paths=$(printf '%s\n' \
  HANDOFF.md \
  docs/rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md \
  docs/rlinf-robotwin-pi0-rltoken/01_CONFIG_PROVENANCE_AND_PRE_SMOKE_PACKET.md \
  docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md |
  LC_ALL=C sort)
test "$actual_paths" = "$expected_paths"

git add -- \
  HANDOFF.md \
  docs/rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md \
  docs/rlinf-robotwin-pi0-rltoken/01_CONFIG_PROVENANCE_AND_PRE_SMOKE_PACKET.md \
  docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md
git diff --cached --check
git commit -m "docs(rlt): lock clean50 and review pre-smoke config"

commit=$(git rev-parse HEAD)
printf 'COMMIT %s\n' "$commit"
timeout 30s git push personal "$branch"
printf '%s\n' 'PUSH_OK'
git status --short
git rev-list --left-right --count HEAD...@{upstream}
