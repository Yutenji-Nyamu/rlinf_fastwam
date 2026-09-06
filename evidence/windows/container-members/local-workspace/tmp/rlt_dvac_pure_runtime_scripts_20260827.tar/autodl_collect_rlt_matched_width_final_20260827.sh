#!/usr/bin/env bash
set -euo pipefail
control_run=/root/autodl-tmp/experiments/rlt_single_gpu_control_matched_width_formal480_20260826_v3
method_run=/root/autodl-tmp/experiments/rlt_single_gpu_success_episode_bc_dvac_matched_width_formal480_20260826_v3
control_rt=/root/autodl-tmp/experiment_exports/rlt_single_gpu_control_matched_width_formal480_20260826_v3/runtime
method_rt=/root/autodl-tmp/experiment_exports/rlt_single_gpu_success_episode_bc_dvac_matched_width_formal480_20260826_v3/runtime
pair_rt=/root/autodl-tmp/experiment_exports/rlt_success_bc_dual_single_gpu_matched_width_formal480_20260826_v3
out=/root/autodl-tmp/experiment_exports/rlt_success_bc_matched_width_final_light_20260827_v1
archive=/root/autodl-tmp/experiment_exports/rlt_success_bc_matched_width_final_raw_20260827_v1.tar.gz

test ! -e "$out"
test ! -e "$archive"
test "$(cat "$control_rt/exit_code.txt")" = 0
test "$(cat "$method_rt/exit_code.txt")" = 0
grep -q 'Global Step:  *480/480' "$control_run/metrics.log"
grep -q 'Global Step:  *480/480' "$method_run/metrics.log"

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
  for name in resolved.yaml exact_command.txt source_head.txt started_at.txt finished_at.txt exit_code.txt foreground.log; do
    cp "$rt/$name" "$out/$kind/$name"
  done
  checkpoint=$(find "$run" -type d -path '*/checkpoints/global_step_480' | head -n 1)
  test -n "$checkpoint"
  find "$checkpoint" -type f -printf '%s %P\n' | sort -k2 >"$out/$kind/checkpoint_480_inventory.txt"
  find "$checkpoint" -type f -name 'complete.json' -print -exec sh -c 'cp "$1" "$2/$(basename "$(dirname "$1")")_complete.json"' sh '{}' "$out/$kind" \;
done

cp "$pair_rt/paired_resources.csv" "$out/pair/paired_resources.csv"
for name in launch_summary.txt resolved_leaf_diff.json; do
  if test -f "$pair_rt/$name"; then cp "$pair_rt/$name" "$out/pair/$name"; fi
done

last_trace=$(find "$method_run" -type f -name 'update_*.npz' -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)
test -n "$last_trace"
cp "$last_trace" "$out/method/latest_complete_trace.npz"

event=$(find "$method_run/tensorboard" -maxdepth 1 -type f -name 'events.out.tfevents.*' | head -n 1)
/root/autodl-tmp/RLinf/.venv/bin/python - "$event" >"$out/method/method_tensorboard_summary.json" <<'PY'
import json
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

acc = EventAccumulator(sys.argv[1], size_guidance={"scalars": 0})
acc.Reload()
wanted = (
    "rlt_dvac/",
    "rlt_dvac_bc/",
    "actor/bc_loss",
    "actor/bc_unweighted_loss",
    "actor/bc_ref_loss",
    "actor/weighted_bc",
    "actor/q_pi",
    "actor/weighted_q",
    "actor/bc_weight",
    "actor/q_weight",
)
result = {}
for tag in sorted(acc.Tags().get("scalars", [])):
    if not any(item in tag for item in wanted):
        continue
    values = acc.Scalars(tag)
    if not values:
        continue
    tail = values[-20:]
    result[tag] = {
        "count": len(values),
        "first_step": values[0].step,
        "latest_step": values[-1].step,
        "latest": values[-1].value,
        "tail20_mean": sum(value.value for value in tail) / len(tail),
    }
print(json.dumps(result, indent=2, sort_keys=True))
PY

{
  date -Is
  echo control_exit="$(cat "$control_rt/exit_code.txt")"
  echo method_exit="$(cat "$method_rt/exit_code.txt")"
  echo control_finished="$(cat "$control_rt/finished_at.txt")"
  echo method_finished="$(cat "$method_rt/finished_at.txt")"
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
