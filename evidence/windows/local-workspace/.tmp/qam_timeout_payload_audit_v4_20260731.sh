#!/usr/bin/env bash
set -euo pipefail

echo "TIME=$(TZ=Asia/Shanghai date '+%F %T %Z')"
hostname
pwd
id -u
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
grep -R -n -E \
  'class .*RoboTwin|def chunk_step|gen_sparse_reward_data|take_action_cnt|step_lim' \
  --include='*.py' \
  "$repo/rlinf/envs" "$repo/rlinf/workers/env" 2>/dev/null |
  head -300 || true
