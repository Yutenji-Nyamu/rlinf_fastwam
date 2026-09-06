#!/usr/bin/env bash
set -eu
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc
cd "$root"
test "$(git branch --show-current)" = codex/sz-pi0-online-bc
test "$(git rev-parse HEAD)" = 5be8747831cd998c7ba0a52d863b5d74e168358e
git diff --check
git diff --stat
git diff -- rlinf/models/embodiment/openpi/openpi_action_model.py examples/embodiment/config/robotwin_adjust_bottle_online_bc_openpi.yaml
test -z "$(git diff --cached --name-only)"
git add -- docs/online_bc.md examples/embodiment/config/robotwin_adjust_bottle_online_bc_openpi.yaml rlinf/models/embodiment/openpi/openpi_action_model.py tests/unit_tests/test_online_bc.py
git commit -m 'Preserve native OpenPI SFT precision and disable BC image augmentation explicitly'
git push personal codex/sz-pi0-online-bc
head=$(git rev-parse HEAD)
remote=$(git ls-remote personal refs/heads/codex/sz-pi0-online-bc | cut -f1)
test "$head" = "$remote"
printf 'VERIFIED_HEAD=%s\n' "$head"
git status --short
