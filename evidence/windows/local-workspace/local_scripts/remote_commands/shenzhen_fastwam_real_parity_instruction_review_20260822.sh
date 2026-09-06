#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711
RT="$WT/third_party/RoboTwin"

printf '%s\n' '=== GENERATE_DESCRIPTION ==='
grep -R -n "def generate_episode_descriptions" "$RT/description" "$RT/script" || true
FILE=$(grep -R -l "def generate_episode_descriptions" "$RT/description" "$RT/script" | sed -n '1p')
test -n "$FILE"
sed -n '1,300p' "$FILE"
printf '%s\n' '=== PLAY_ONCE_ACTION_EVIDENCE ==='
grep -R -n -E "def play_once|take_action\(" "$RT/envs/adjust_bottle.py" "$RT/envs/_base_task.py" "$RT/envs" 2>/dev/null | sed -n '1,240p' || true
printf '%s\n' '=== TASK_CONFIG_LOCATORS ==='
find /data/chenyiteng -maxdepth 8 -type f -path '*/task_config/demo_clean.yml' -print 2>/dev/null || true
