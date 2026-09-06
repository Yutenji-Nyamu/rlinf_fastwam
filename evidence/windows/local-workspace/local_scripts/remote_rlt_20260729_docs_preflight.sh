#!/usr/bin/env bash
set -euo pipefail

worktree=/root/autodl-tmp/RLinf_rlt_pi0_robotwin

date -Is
git -C "$worktree" status --short
git -C "$worktree" branch --show-current
git -C "$worktree" rev-parse HEAD
git -C "$worktree" rev-list --left-right --count HEAD...@{upstream}

for path in \
  HANDOFF.md \
  docs/rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md \
  docs/rlinf-robotwin-pi0-rltoken/01_CONFIG_PROVENANCE_AND_PRE_SMOKE_PACKET.md \
  docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md
do
  if git -C "$worktree" ls-files --error-unmatch "$path" >/dev/null 2>&1; then
    printf 'TRACKED %s\n' "$path"
  else
    printf 'NOT_TRACKED %s\n' "$path"
  fi
done
