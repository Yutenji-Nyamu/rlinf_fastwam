#!/usr/bin/env bash
set -euo pipefail

FW=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711
VRT="$FW/third_party/RoboTwin"
PYTHON=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128/bin/python
CFG=fw-sz-400-20260822_042858_demo_clean_1ep

cd "$VRT"
printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
printf 'command=CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 %q -X faulthandler script/collect_data.py adjust_bottle %q\n' "$PYTHON" "$CFG"
CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 \
  timeout --signal=TERM --kill-after=30s 300s \
  "$PYTHON" -X faulthandler script/collect_data.py adjust_bottle "$CFG"
