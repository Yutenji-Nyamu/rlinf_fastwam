#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
cd "$WT"
git diff --check
git add examples/embodiment/config/robotwin_move_stapler_pad_grpo_openpi_pi05_sidney.yaml
git commit -m "fix(robotwin): preserve Sidney 400-step task horizon"
git push personal codex/sz-sidney-pi05-current-rlinf
git status --short
git rev-parse HEAD
git ls-remote personal refs/heads/codex/sz-sidney-pi05-current-rlinf
