#!/usr/bin/env bash
set -euo pipefail
OUTPUT=/data/chenyiteng/results/dvac-observation/analysis/sz-dvac-fastwam-move-stapler-p2-phase-v1
for name in query_metrics.csv fastwam_action_frame_metrics.csv phase_episode_metrics.csv phase_summary.csv storyboard_index.csv; do
  printf 'FILE=%s\n' "$name"
  sed -n '1,3p' "$OUTPUT/$name"
done
