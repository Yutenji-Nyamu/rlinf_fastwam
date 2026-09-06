#!/usr/bin/env bash
set -euo pipefail

CANON=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711
cd "$CANON/third_party/RoboTwin"
echo "TIME_UTC=$(date -u +%FT%TZ)"
grep -nE '^adjust_bottle:' task_config/_eval_step_limit.yml
grep -nE '^(seed|mixed_precision|eval_num_inference_steps):' "$CANON/configs/train.yaml" "$CANON/configs/sim_robotwin.yaml" 2>/dev/null || true
echo FASTWAM_DVAC_P0_BUDGET_SOURCE_PROBE_OK
