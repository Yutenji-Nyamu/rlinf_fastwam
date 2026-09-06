#!/usr/bin/env bash
set -euo pipefail

RUN=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_apply_formal_100step_2gpu16env_20260821
EXP="$RUN/idea2_dvac_apply_formal_100step_2gpu16env_20260821"
RUNTIME=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_apply_formal_100step_2gpu16env_20260821
SOURCE=/root/autodl-tmp/RLinf_idea2_dvac_train
STAGE=/root/autodl-tmp/idea2_dvac_v1_formal_stop_g54_closeout_20260821
ARCHIVE=/root/autodl-tmp/idea2_dvac_v1_formal_stop_g54_closeout_20260821.tar.gz

test ! -e "$STAGE"
test ! -e "$ARCHIVE"
mkdir -p "$STAGE/config" "$STAGE/runtime" "$STAGE/run/dvac_train" "$STAGE/analysis"

cp "$SOURCE/examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_dvac_train_100step_formal.yaml" \
  "$STAGE/config/source_config.yaml"
cp "$RUN/metrics.log" "$STAGE/run/metrics.log"
cp -a "$RUN/control_trace" "$STAGE/run/control_trace"

for name in launch_command.txt launch_started_at.txt stop_requested_at.txt terminate_requested_at.txt launch_finished_at.txt driver.exitcode driver.log wrapper.log observer.log; do
  test -f "$RUNTIME/$name" && cp "$RUNTIME/$name" "$STAGE/runtime/$name"
done
cp "$RUNTIME/resource_monitor/observer_exit.txt" "$STAGE/runtime/observer_exit.txt"

for rank in actor_rank00 actor_rank01; do
  mkdir -p "$STAGE/run/dvac_train/$rank"
  for name in run_manifest.json runner_step_metrics.csv rolling_stats_state.json rollout_step0000.npz rollout_step0001.npz rollout_step0053.npz; do
    cp "$RUN/dvac_train/$rank/$name" "$STAGE/run/dvac_train/$rank/$name"
  done
done

awk -F, '
  NR==1 {next}
  {
    gpu=$3+0;
    if (($4+0)>gpu_peak[gpu]) gpu_peak[gpu]=$4+0;
    if (($10+0)>cgroup_peak) cgroup_peak=$10+0;
    if (($14+0)>event_high) event_high=$14+0;
    if (($15+0)>event_max) event_max=$15+0;
    if (($16+0)>event_oom) event_oom=$16+0;
    if (($17+0)>event_oom_kill) event_oom_kill=$17+0;
    last_timestamp=$1; last_elapsed=$2;
  }
  END {
    printf "last_timestamp=%s\nlast_elapsed_s=%s\n", last_timestamp, last_elapsed;
    printf "gpu0_peak_mib=%d\ngpu1_peak_mib=%d\n", gpu_peak[0], gpu_peak[1];
    printf "cgroup_peak_bytes=%.0f\n", cgroup_peak;
    printf "event_high=%d\nevent_max=%d\nevent_oom=%d\nevent_oom_kill=%d\n", event_high, event_max, event_oom, event_oom_kill;
  }
' "$RUNTIME/resource_monitor/resources.csv" > "$STAGE/analysis/RESOURCE_SUMMARY.txt"

checkpoint="$EXP/checkpoints/global_step_50"
{
  echo "checkpoint_name=global_step_50"
  echo "checkpoint_path=$checkpoint"
  echo "checkpoint_files=$(find "$checkpoint" -type f | wc -l)"
  echo "checkpoint_bytes=$(du -sb "$checkpoint" | awk '{print $1}')"
  echo "checkpoint_partials=$(find "$checkpoint" -type f -name '*.partial' | wc -l)"
  echo "included_in_archive=false"
} > "$STAGE/CHECKPOINT_POINTER.txt"

tar -C /root/autodl-tmp -czf "$ARCHIVE" "$(basename "$STAGE")"
echo "STAGE_BYTES=$(du -sb "$STAGE" | awk '{print $1}')"
echo "ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE")"
echo "ARCHIVE=$ARCHIVE"
