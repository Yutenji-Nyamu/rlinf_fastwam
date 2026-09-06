#!/usr/bin/env bash
set -euo pipefail

echo "TIME=$(TZ=Asia/Shanghai date '+%F %T %Z')"
hostname
pwd
id -u

root=/root/autodl-tmp/RoboTwin_RLinf
echo "ROBOWTIN_GIT"
git -C "$root" rev-parse HEAD 2>/dev/null || true
git -C "$root" status --short 2>/dev/null || true

echo "CHUNK_STEP_DEFINITIONS"
rg -n --glob '*.py' \
  'def chunk_step|truncat|terminated|_elapsed_steps|max_episode_steps|final_observation' \
  "$root" |
  head -300

echo "RLINF_TRANSITION_SELECTION"
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
sed -n '450,525p' "$repo/rlinf/workers/env/env_worker.py"
sed -n '1115,1150p' "$repo/rlinf/workers/env/env_worker.py"
