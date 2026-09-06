#!/usr/bin/env bash
set -u
out=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-pi0-robotwin-7d07a421/evidence/completed_runs_20260822_20260903
grep -ERin --binary-files=without-match \
  -e '[REDACTED]' -e '123456' \
  -e 'password[[:space:]]*[:=]' -e 'api[_-]\?key[[:space:]]*[:=]' \
  -e 'authorization:[[:space:]]*bearer' -e 'hf_[A-Za-z0-9]\{20,\}' -e 'github_pat_' \
  -e 'BEGIN \(RSA\|OPENSSH\|EC\) PRIVATE KEY' "$out" | sed -n '1,30p' || true
