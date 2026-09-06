#!/usr/bin/env bash
set -euo pipefail
find /data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab -maxdepth 2 -type f -printf '%P\n' | sort
echo '=== norm example ==='
sed -n '1,100p' /data/chenyiteng/models/rlinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle@fa8df6ed/physical-intelligence/robotwin/norm_stats.json
echo '=== relevant full config ==='
sed -n '1,260p' /data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf/examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_pi05.yaml
