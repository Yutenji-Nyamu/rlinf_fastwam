#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128/bin/python
SCRIPT=/data/chenyiteng/results/dvac-observation/packets/sz-dvac-analysis-v1/render_shenzhen_dvac_cross_task_20260823.py
ROOT=/data/chenyiteng/results/dvac-observation/analysis/sz-dvac-pi0-fixed64-fastwam-four-tasks-v1
OUTPUT="$ROOT/figures/cross_task_outcome_and_position_summary.png"

test -x "$PYTHON"
test -f "$SCRIPT"
test -d "$ROOT"
test ! -e "$OUTPUT"
CUDA_VISIBLE_DEVICES='' "$PYTHON" "$SCRIPT" --analysis-root "$ROOT"
stat -c 'FIGURE_BYTES=%s' "$OUTPUT"
sha256sum "$OUTPUT"
printf '%s\n' 'SZ_DVAC_CROSS_TASK_FIGURE_OK'
