#!/usr/bin/env bash
set -u

run_dir=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_downweight_smoke_2step_2gpu16env_20260821

echo METRICS_TAIL
tail -n 80 "$run_dir/metrics.log" 2>/dev/null || true

for rank in 00 01; do
  shard="$run_dir/dvac_train/actor_rank${rank}"
  echo "RANK_${rank}_CSV"
  cat "$shard/runner_step_metrics.csv" 2>/dev/null || true
  echo "RANK_${rank}_ROLLING"
  cat "$shard/rolling_stats_state.json" 2>/dev/null || true
done
