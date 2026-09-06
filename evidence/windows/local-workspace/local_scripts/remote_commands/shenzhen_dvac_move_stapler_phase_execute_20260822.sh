#!/usr/bin/env bash
set -euo pipefail

PACKET=/data/chenyiteng/results/dvac-observation/packets/sz-dvac-analysis-v1
ANALYZER="$PACKET/analyze_shenzhen_dvac_observation.py"
PHASE="$PACKET/fastwam_move_stapler_phase_annotations_v1.csv"
SOURCE=/data/chenyiteng/results/dvac-observation/fastwam-move_stapler_pad-p2-16ep-c63dc9b5-v1
VIDEO=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711/evaluate_results/robotwin/robotwin_uncond_3cam_384/fastwam-move_stapler_pad-p2-16ep-c63dc9b5-v1
OUTPUT=/data/chenyiteng/results/dvac-observation/analysis/sz-dvac-fastwam-move-stapler-p2-phase-v1
PY=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128/bin/python

test "$(sha256sum "$ANALYZER" | awk '{print $1}')" = 009b4ea41ab9ebc974b778e46156ee5fc4f0bf843e0fd703d3051715675ff152
test "$(sha256sum "$PHASE" | awk '{print $1}')" = a949e73c060aef534e37c9dcced1306cb17b684a5a8773532dcbb8b5e8dc8afc
test "$(wc -l < "$PHASE")" -eq 17
test ! -e "$OUTPUT"

export CUDA_VISIBLE_DEVICES=
export PYTHONUNBUFFERED=1
printf 'START=%s\n' "$(date --iso-8601=seconds)"
/usr/bin/time -f 'TIME_WALL=%e TIME_USER=%U TIME_SYS=%S MAX_RSS_KIB=%M' \
  "$PY" "$ANALYZER" \
  --fastwam-source "$SOURCE" \
  --phase-annotations "$PHASE" \
  --video-root "$VIDEO" \
  --output "$OUTPUT"
printf 'END=%s\nANALYSIS_EXECUTE_OK\n' "$(date --iso-8601=seconds)"
