#!/usr/bin/env bash
set -euo pipefail

control_run=/root/autodl-tmp/experiments/rlt_single_gpu_control_matched_width_formal480_20260826_v3
method_run=/root/autodl-tmp/experiments/rlt_single_gpu_success_episode_bc_dvac_matched_width_formal480_20260826_v3
control_rt=/root/autodl-tmp/experiment_exports/rlt_single_gpu_control_matched_width_formal480_20260826_v3/runtime
method_rt=/root/autodl-tmp/experiment_exports/rlt_single_gpu_success_episode_bc_dvac_matched_width_formal480_20260826_v3/runtime
pair_rt=/root/autodl-tmp/experiment_exports/rlt_success_bc_dual_single_gpu_matched_width_formal480_20260826_v3
out=/root/autodl-tmp/experiment_exports/rlt_success_bc_matched_width_stopped_light_20260827_v1
archive=/root/autodl-tmp/experiment_exports/rlt_success_bc_matched_width_stopped_raw_20260827_v1.tar.gz

test ! -e "$out"
test ! -e "$archive"
for rt in "$control_rt" "$method_rt"; do
  pid=$(cat "$rt/wrapper.pid")
  ! kill -0 "$pid" 2>/dev/null
done
ray_pid=$(cat "$pair_rt/ray_head.pid")
! kill -0 "$ray_pid" 2>/dev/null

mkdir -p "$out/control" "$out/method" "$out/pair"
for spec in control:"$control_run":"$control_rt" method:"$method_run":"$method_rt"; do
  kind=${spec%%:*}
  rest=${spec#*:}
  run=${rest%%:*}
  rt=${rest#*:}
  cp "$run/metrics.log" "$out/$kind/metrics.log"
  cp "$run/tensorboard/config.yaml" "$out/$kind/tensorboard_config.yaml"
  event=$(find "$run/tensorboard" -maxdepth 1 -type f -name 'events.out.tfevents.*' | head -n 1)
  cp "$event" "$out/$kind/$(basename "$event")"
  for name in resolved.yaml exact_command.txt source_head.txt started_at.txt compose_exit_code.txt resolved.sha256; do
    test -f "$rt/$name" && cp "$rt/$name" "$out/$kind/$name"
  done
  tail -n 1500 "$rt/foreground.log" >"$out/$kind/foreground_tail_1500.log"
  latest_checkpoint=$(find "$run" -type d -path '*/checkpoints/global_step_*' -printf '%f %p\n' \
    | awk '{step=$1; sub("global_step_", "", step); print step, $2}' | sort -n | tail -n 1 | awk '{print $2}')
  test -n "$latest_checkpoint"
  checkpoint_step=${latest_checkpoint##*global_step_}
  printf '%s\n' "$checkpoint_step" >"$out/$kind/latest_checkpoint_step.txt"
  find "$latest_checkpoint" -type f -printf '%s %P\n' | sort -k2 >"$out/$kind/latest_checkpoint_inventory.txt"
  find "$latest_checkpoint" -type f -name 'complete.json' -print -exec sh -c \
    'dest=$1; shift; for source do name=$(basename "$(dirname "$source")"); cp "$source" "$dest/${name}_complete.json"; done' \
    sh "$out/$kind" '{}' +
done

cp "$pair_rt/paired_resources.csv" "$out/pair/paired_resources.csv"
for name in launch_summary.txt resolved_leaf_diff.json user_stop_requested_at.txt; do
  test -f "$pair_rt/$name" && cp "$pair_rt/$name" "$out/pair/$name"
done

last_trace=$(find "$method_run" -type f -name 'update_*.npz' -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)
test -n "$last_trace"
cp "$last_trace" "$out/method/latest_complete_trace.npz"

{
  echo "status=user-stopped-near-end"
  echo "requested_at=$(cat "$pair_rt/user_stop_requested_at.txt")"
  echo "control_last_checkpoint=$(cat "$out/control/latest_checkpoint_step.txt")"
  echo "method_last_checkpoint=$(cat "$out/method/latest_checkpoint_step.txt")"
  echo fatal_counts
  for rt in "$control_rt" "$method_rt"; do
    printf '%s ' "$rt"
    for pattern in 'CUDA out of memory' 'OutOfMemoryError' 'WorkerCrashedError' 'ActorDiedError' 'NCCL error'; do
      printf '%s=%s ' "$pattern" "$(grep -cF "$pattern" "$rt/foreground.log" || true)"
    done
    echo
  done
  echo memory_events
  cat /sys/fs/cgroup/memory.events
  echo gpu
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
} >"$out/final_state.txt"

find "$out" -type f -printf '%s %P\n' | sort -k2 >"$out/archive_member_manifest.txt"
tar -czf "$archive" -C "$(dirname "$out")" "$(basename "$out")"
sha256sum "$archive"
du -h "$archive"
cat "$out/final_state.txt"
