#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711
RT="$WT/third_party/RoboTwin"

printf '%s\n' '=== INFER_ACTION_CONTEXT ==='
sed -n '930,1140p' "$WT/src/fastwam/models/wan22/fastwam.py"
printf '%s\n' '=== SCHEDULER_CONTEXT ==='
grep -R -n -E "class .*Scheduler|def build_inference_schedule|def step\(" "$WT/src" "$WT/experiments" || true
SCHED=$(grep -R -l "def build_inference_schedule" "$WT/src" | sed -n '1p')
test -n "$SCHED"
sed -n '1,260p' "$SCHED"
printf '%s\n' '=== POLICY_CONTEXT ==='
grep -n -E "def get_model|def _build_robotwin_image_tensor|def _normalize_state|def _denormalize_action|class .*Policy|def update_obs|infer_action\(" "$WT/experiments/robotwin/fastwam_policy/deploy_policy.py" || true
sed -n '1,420p' "$WT/experiments/robotwin/fastwam_policy/deploy_policy.py"
printf '%s\n' '=== ROBOTWIN_OFFICIAL_EVAL_CONTEXT ==='
grep -n -E "setup_demo|play_once|generate_episode_descriptions|set_instruction|get_obs|take_action|close_env|plan_success|check_success" "$RT/eval_policy.py" || true
sed -n '110,330p' "$RT/eval_policy.py"
printf '%s\n' '=== CONFIG_CONTEXT ==='
sed -n '1,260p' "$WT/configs/sim_robotwin.yaml"
sed -n '1,220p' "$RT/task_config/demo_clean.yml"
