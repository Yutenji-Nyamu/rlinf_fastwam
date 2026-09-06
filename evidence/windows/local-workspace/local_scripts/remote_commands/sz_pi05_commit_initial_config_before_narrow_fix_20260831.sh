#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-robotwin-rl
test "$(git -C "$WT" rev-parse HEAD)" = 74617ced87d64045ab6850d0efd90956a494af66
test "$(git -C "$WT" status --short | wc -l)" -eq 2
git -C "$WT" add \
  examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_pi05.yaml \
  examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_pi05_dvac_action_adv.yaml
git -C "$WT" commit -m 'feat(embodiment): add pi0.5 RoboTwin GRPO configs'
git -C "$WT" rev-parse HEAD
