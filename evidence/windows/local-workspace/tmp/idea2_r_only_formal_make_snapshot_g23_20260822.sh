#!/usr/bin/env bash
set -euo pipefail

base=/root/autodl-tmp
stem=idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822
run_rel=idea2_dvac_train_runs/$stem
runtime_rel=idea2_dvac_train_runtime/$stem
archive="$base/$runtime_rel/current_snapshot_g23_20260822.tar.gz"

test -f "$base/$run_rel/metrics.log"
test -f "$base/$run_rel/dvac_train/actor_rank00/rollout_step0022.npz"
test -f "$base/$run_rel/dvac_train/actor_rank01/rollout_step0022.npz"

cp "$base/$run_rel/metrics.log" "$base/$runtime_rel/snapshot_metrics_g23.log"
cp "$base/$runtime_rel/driver.log" "$base/$runtime_rel/snapshot_driver_g23.log"
cp "$base/$runtime_rel/resource_monitor/resources.csv" "$base/$runtime_rel/snapshot_resources_g23.csv"
cp "$base/$runtime_rel/resource_monitor/process_rss.tsv" "$base/$runtime_rel/snapshot_process_rss_g23.tsv"

tar -czf "$archive" -C "$base" \
  "$run_rel/dvac_train/actor_rank00/run_manifest.json" \
  "$run_rel/dvac_train/actor_rank00/runner_step_metrics.csv" \
  "$run_rel/dvac_train/actor_rank00/rolling_stats_state.json" \
  "$run_rel/dvac_train/actor_rank00/rollout_step0022.npz" \
  "$run_rel/dvac_train/actor_rank01/run_manifest.json" \
  "$run_rel/dvac_train/actor_rank01/runner_step_metrics.csv" \
  "$run_rel/dvac_train/actor_rank01/rolling_stats_state.json" \
  "$run_rel/dvac_train/actor_rank01/rollout_step0022.npz" \
  "$run_rel/control_trace" \
  "$runtime_rel/resolved_config.yaml" \
  "$runtime_rel/launch_command.txt" \
  "$runtime_rel/launch_started_at.txt" \
  "$runtime_rel/snapshot_metrics_g23.log" \
  "$runtime_rel/snapshot_driver_g23.log" \
  "$runtime_rel/snapshot_resources_g23.csv" \
  "$runtime_rel/snapshot_process_rss_g23.tsv"

ls -lh "$archive"
tar -tzf "$archive" | wc -l
