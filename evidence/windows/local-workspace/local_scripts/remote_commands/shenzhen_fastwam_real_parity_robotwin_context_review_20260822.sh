#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711
RT="$WT/third_party/RoboTwin"

printf '%s\n' '=== ROBOTWIN_TREE ==='
ls -la "$RT"
find "$RT" -maxdepth 3 -type l -print -exec readlink -f {} \; 2>/dev/null || true
printf '%s\n' '=== EVAL_POLICY_RELEVANT ==='
grep -n -E "setup_demo|play_once|generate_episode_descriptions|set_instruction|get_obs|take_action|close_env|plan_success|check_success|class_decorator" "$RT/script/eval_policy.py" || true
sed -n '1,360p' "$RT/script/eval_policy.py"
printf '%s\n' '=== TASK_CONFIG ==='
sed -n '1,240p' "$RT/task_config/demo_clean.yml"
printf '%s\n' '=== FASTWAM_EVAL_ENTRY ==='
grep -R -n -E "eval_policy.py|demo_clean|reset_seed|unseen|seed" "$WT/experiments/robotwin" "$WT/scripts" 2>/dev/null || true
