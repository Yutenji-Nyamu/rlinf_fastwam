#!/usr/bin/env bash
set -euo pipefail
OUTPUT=/data/chenyiteng/results/dvac-observation/analysis/sz-dvac-fastwam-move-stapler-p2-phase-v1
for name in outcome_summary.csv phase_summary.csv query_metrics.csv L_sensitivity.csv episode_metrics.csv; do
  printf 'FILE=%s\n' "$name"
  sed -n '1,8p' "$OUTPUT/$name"
done
