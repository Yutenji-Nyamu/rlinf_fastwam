#!/usr/bin/env bash
set -u
base=/root/autodl-tmp/RLinf

echo '=== historical-worktree-status ==='
for wt in \
  /root/autodl-tmp/RLinf_fastwam_rlinf \
  /root/autodl-tmp/RLinf_qam_pi0_robotwin \
  /root/autodl-tmp/RLinf_rlt_pi0_robotwin
do
  printf '%s\n' "$wt"
  git -C "$wt" status --short --branch
done

echo '=== ppo-robotwin-config ==='
nl -ba "$base/examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi.yaml" \
  | sed -n '39,68p;124,176p'

echo '=== pi0-model-defaults ==='
nl -ba "$base/examples/embodiment/config/model/pi0.yaml" | sed -n '1,35p'

echo '=== pi0-sde-config-and-schedule ==='
f="$base/rlinf/models/embodiment/openpi/openpi_action_model.py"
nl -ba "$f" | sed -n '43,72p;972,1058p;1075,1153p;1622,1638p'
