#!/usr/bin/env bash
set -euo pipefail

root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-observe-7d07a421
cd "$root"
printf '%s  %s\n' \
  706806020bd1fe6d98d1c6489bd91e5d2ab7968fed2e7cb7761262a602fb91ce \
  tests/unit_tests/test_dvac_telemetry.py | sha256sum --check
git add -- tests/unit_tests/test_dvac_telemetry.py
test "$(git diff --cached --name-only | wc -l)" -eq 6
test -z "$(git diff --name-only)"
git diff --cached --check
git status --short
