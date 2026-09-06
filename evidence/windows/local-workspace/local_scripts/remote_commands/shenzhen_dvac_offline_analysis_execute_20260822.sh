#!/usr/bin/env bash
set -euo pipefail

SCRIPT=/data/chenyiteng/results/dvac-observation/packets/sz-dvac-analysis-v1/analyze_shenzhen_dvac_observation.py
PYTHON=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128/bin/python
PI0_SOURCE=/data/chenyiteng/results/dvac-observation/pi0-adjust_bottle-fixed64-800baf80-v1
FASTWAM_SOURCE=/data/chenyiteng/results/dvac-observation/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2
VIDEO_ROOT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711/evaluate_results/robotwin/robotwin_uncond_3cam_384/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2
OUTPUT=/data/chenyiteng/results/dvac-observation/analysis/sz-dvac-pi0-fixed64-fastwam-adjust-p1-v1

test -f "$SCRIPT"
test -d "$PI0_SOURCE"
test -d "$FASTWAM_SOURCE"
test -d "$VIDEO_ROOT"
test ! -e "$OUTPUT"
printf 'ANALYSIS_START=%s\n' "$(date -Is)"
/usr/bin/time -f 'ANALYSIS_TIME wall=%e user=%U sys=%S maxrss_kib=%M exit=%x' \
  "$PYTHON" "$SCRIPT" \
    --pi0-source "$PI0_SOURCE" \
    --fastwam-source "$FASTWAM_SOURCE" \
    --video-root "$VIDEO_ROOT" \
    --output "$OUTPUT"
printf 'ANALYSIS_END=%s\nANALYSIS_EXECUTE_OK\n' "$(date -Is)"
