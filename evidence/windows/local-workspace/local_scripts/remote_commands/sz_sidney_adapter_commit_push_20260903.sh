#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
cd "$WT"
git add \
  rlinf/utils/ckpt_convertor/openpi/lerobot_pi05_to_openpi_rlinf.py \
  rlinf/utils/ckpt_convertor/openpi/convert.py \
  rlinf/models/embodiment/openpi/dataconfig/robotwin_aloha_dataconfig.py \
  rlinf/models/embodiment/openpi/dataconfig/__init__.py \
  examples/embodiment/config/robotwin_move_stapler_pad_grpo_openpi_pi05_sidney.yaml \
  tests/unit_tests/test_lerobot_pi05_importer.py
git commit -m "feat(openpi): import Sidney RoboTwin pi0.5 checkpoint"
git push -u personal codex/sz-sidney-pi05-current-rlinf
git status --short
git show --stat --oneline --summary HEAD
git rev-parse HEAD
git ls-remote personal refs/heads/codex/sz-sidney-pi05-current-rlinf
