#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711

printf '%s\n' '=== Z_HELPER ==='
sed -n '1,320p' "$WT/experiments/robotwin/fastwam_policy/dvac_telemetry.py"
printf '%s\n' '=== CURRENT EVAL CONFIGS ==='
grep -R -n -E "eval_num_episodes|instruction_type|seed:|task_config|adjust_bottle|robotwin_uncond_3cam_384" "$WT/experiments/robotwin" "$WT/configs" 2>/dev/null || true
printf '%s\n' '=== PLAY_ONCE_BODY ==='
sed -n '1,150p' "$WT/third_party/RoboTwin/envs/adjust_bottle.py"
