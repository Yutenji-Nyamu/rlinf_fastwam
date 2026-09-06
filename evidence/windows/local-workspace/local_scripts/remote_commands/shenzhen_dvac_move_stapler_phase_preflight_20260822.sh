#!/usr/bin/env bash
set -euo pipefail

PACKET=/data/chenyiteng/results/dvac-observation/packets/sz-dvac-analysis-v1
ANALYZER="$PACKET/analyze_shenzhen_dvac_observation.py"
PHASE="$PACKET/fastwam_move_stapler_phase_annotations_v1.csv"
SOURCE=/data/chenyiteng/results/dvac-observation/fastwam-move_stapler_pad-p2-16ep-c63dc9b5-v1
VIDEO=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711/evaluate_results/robotwin/robotwin_uncond_3cam_384/fastwam-move_stapler_pad-p2-16ep-c63dc9b5-v1
OUTPUT=/data/chenyiteng/results/dvac-observation/analysis/sz-dvac-fastwam-move-stapler-p2-phase-v1
PY=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128/bin/python

test -d "$PACKET"
test -s "$ANALYZER"
test "$(sha256sum "$ANALYZER" | awk '{print $1}')" = 009b4ea41ab9ebc974b778e46156ee5fc4f0bf843e0fd703d3051715675ff152
test ! -e "$PHASE"
test -s "$SOURCE/episodes.csv"
test -s "$SOURCE/queries.csv"
test -d "$VIDEO"
test ! -e "$OUTPUT"
test -x "$PY"
printf 'PHASE_TARGET=%s\nOUTPUT=%s\nPREFLIGHT_OK\n' "$PHASE" "$OUTPUT"
