#!/usr/bin/env bash
set -euo pipefail

echo "TIME=$(TZ=Asia/Shanghai date '+%F %T %Z')"
hostname
pwd
id -u

root=/root/autodl-tmp/RoboTwin_RLinf
echo "CANDIDATE_FILES"
find "$root" -type f -name '*.py' \
  \( -path '*/envs/*' -o -path '*/script/*' -o -path '*/policy/*' \) \
  -print |
  grep -E 'robotwin|env|wrapper|base|task' |
  head -120

echo "TIMEOUT_SYMBOLS"
grep -R -n -E \
  'def chunk_step|truncat|terminated|_elapsed_steps|max_episode_steps|final_observation' \
  --include='*.py' \
  "$root/envs" "$root/script" 2>/dev/null |
  head -300 || true

echo "RLINF_TRANSITION_SELECTION"
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
sed -n '450,525p' "$repo/rlinf/workers/env/env_worker.py"
sed -n '1115,1150p' "$repo/rlinf/workers/env/env_worker.py"
