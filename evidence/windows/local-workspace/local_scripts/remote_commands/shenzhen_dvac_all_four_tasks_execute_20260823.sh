#!/usr/bin/env bash
set -euo pipefail

ANALYZER=/data/chenyiteng/results/dvac-observation/packets/sz-dvac-analysis-v1/analyze_shenzhen_dvac_observation.py
PYTHON=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128/bin/python
PHASE=/data/chenyiteng/results/dvac-observation/packets/sz-dvac-analysis-v1/fastwam_move_stapler_phase_annotations_v1.csv
OUTPUT=/data/chenyiteng/results/dvac-observation/analysis/sz-dvac-pi0-fixed64-fastwam-four-tasks-v1

PI0=/data/chenyiteng/results/dvac-observation/pi0-adjust_bottle-fixed64-800baf80-v1
FW_ADJUST=/data/chenyiteng/results/dvac-observation/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2
FW_MOVE=/data/chenyiteng/results/dvac-observation/fastwam-move_stapler_pad-p2-16ep-c63dc9b5-v1
FW_TURN=/data/chenyiteng/results/dvac-observation/fastwam-turn_switch-p2-16ep-c63dc9b5-v1
FW_PICK=/data/chenyiteng/results/dvac-observation/fastwam-pick_diverse_bottles-p2-16ep-c63dc9b5-v1

VIDEO_BASE=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711/evaluate_results/robotwin/robotwin_uncond_3cam_384

test -x "$PYTHON"
test -f "$ANALYZER"
test -f "$PHASE"
for source in "$PI0" "$FW_ADJUST" "$FW_MOVE" "$FW_TURN" "$FW_PICK"; do
  test -d "$source"
done
test ! -e "$OUTPUT"

CUDA_VISIBLE_DEVICES='' /usr/bin/time -v "$PYTHON" "$ANALYZER" \
  --pi0-source "$PI0" \
  --fastwam-source "$FW_ADJUST" \
  --fastwam-source "$FW_MOVE" \
  --fastwam-source "$FW_TURN" \
  --fastwam-source "$FW_PICK" \
  --phase-annotations "$PHASE" \
  --video-root "$VIDEO_BASE/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2" \
  --video-root "$VIDEO_BASE/fastwam-move_stapler_pad-p2-16ep-c63dc9b5-v1" \
  --video-root "$VIDEO_BASE/fastwam-turn_switch-p2-16ep-c63dc9b5-v1" \
  --video-root "$VIDEO_BASE/fastwam-pick_diverse_bottles-p2-16ep-c63dc9b5-v1" \
  --output "$OUTPUT"

test -f "$OUTPUT/analysis_summary.json"
printf 'OUTPUT_FILES=%s\n' "$(find "$OUTPUT" -type f | wc -l)"
du -sb "$OUTPUT"
printf '%s\n' 'SZ_DVAC_ALL_FOUR_TASKS_ANALYSIS_OK'
