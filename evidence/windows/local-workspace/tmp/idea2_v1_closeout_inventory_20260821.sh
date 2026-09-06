#!/usr/bin/env bash
set -euo pipefail

RUN=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_apply_formal_100step_2gpu16env_20260821
RUNTIME=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_apply_formal_100step_2gpu16env_20260821

for path in \
  "$RUN/metrics.log" \
  "$RUNTIME/launch_command.txt" \
  "$RUNTIME/launch_started_at.txt" \
  "$RUNTIME/stop_requested_at.txt" \
  "$RUNTIME/terminate_requested_at.txt" \
  "$RUNTIME/launch_finished_at.txt" \
  "$RUNTIME/driver.exitcode" \
  "$RUNTIME/driver.log" \
  "$RUNTIME/wrapper.log" \
  "$RUNTIME/observer.log"; do
  test -f "$path" && stat -c '%s %n' "$path"
done

for rank in actor_rank00 actor_rank01; do
  dir="$RUN/dvac_train/$rank"
  for name in run_manifest.json runner_step_metrics.csv rolling_stats_state.json rollout_step0000.npz rollout_step0001.npz rollout_step0053.npz; do
    path="$dir/$name"
    test -f "$path" && stat -c '%s %n' "$path"
  done
done

find "$RUN/control_trace" -type f -printf '%s %p\n' | sort -n
echo "CONTROL_TRACE_BYTES=$(du -sb "$RUN/control_trace" | awk '{print $1}')"
echo "RESOURCE_RAW_BYTES=$(du -sb "$RUNTIME/resource_monitor" | awk '{print $1}')"
find "$RUNTIME/resource_monitor" -maxdepth 1 -type f -printf '%s %f\n' | sort -n
